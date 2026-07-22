use std::{
    borrow::Cow,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    error::Error,
    marker::PhantomData,
    rc::Rc,
    str::FromStr,
};

use heed::{
    BytesDecode, BytesEncode, Database, DatabaseFlags, Env, EnvOpenOptions, Error as HeedError,
    WithoutTls,
    byteorder::NativeEndian,
    types::{Lazy, LazyDecode, SerdeJson, U64},
};
use pyo3::{prelude::*, types::PyString};
use regex::regex;
use serde::{Deserialize, Serialize};

use crate::{
    size_sketch::{
        HighPrecisionSizeSketch, LowPrecisionSizeSketch, MediumPrecisionSizeSketch, SizeSketch,
    },
    summed_radix_tree::SummedRadixTree,
    tarjan::{self, GraphSCCVisitor},
};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[pyclass(frozen, skip_from_py_object, get_all)]
/// Lightweight object metadata used in lists and graph paths.
pub struct ObjectSummary {
    pub id: Id,
    pub r#type: String,
    pub value: Option<String>,
    pub size: u64,
    pub subtree_size: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct RawObjectRecord {
    id: Id,
    r#type: String,
    references: Vec<Id>,
    value: Option<String>,
    size: u64,
    subtree_size: Option<u64>,
}

impl RawObjectRecord {
    fn expect_subtree_size(&self) -> anyhow::Result<u64> {
        self.subtree_size
            .ok_or_else(|| anyhow::anyhow!("No subtree size in record for object {}", self.id))
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[pyclass(frozen, skip_from_py_object, get_all)]
/// Full object record including references and referrers.
pub struct ObjectRecord {
    pub id: Id,
    pub r#type: String,
    pub references: Vec<ObjectSummary>,
    pub referrers: Vec<ObjectSummary>,
    pub value: Option<String>,
    pub size: u64,
    pub subtree_size: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SizeIndexEntry {
    size: u64,
    obj_id: u64,
}

impl Ord for SizeIndexEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort in reverse size order, then by obj_id ascending to break ties, so we can efficiently page through largest objects in LMDB.
        other
            .size
            .cmp(&self.size)
            .then_with(|| self.obj_id.cmp(&other.obj_id))
    }
}

impl PartialOrd for SizeIndexEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl BytesEncode<'_> for SizeIndexEntry {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        let mut buf = [0u8; 16];
        // Pack by size descending, then by obj_id ascending, so we can sort by size in LMDB.
        // This enables efficient retrieval of largest objects per type.
        // Use BE bytes so sorting is lexical
        buf[..8].copy_from_slice(&(u64::MAX - item.size).to_be_bytes());
        buf[8..].copy_from_slice(&(item.obj_id).to_be_bytes());
        Ok(Cow::Owned(buf.to_vec()))
    }
}

impl BytesDecode<'_> for SizeIndexEntry {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        if bytes.len() != 16 {
            return Err(format!(
                "Invalid byte length for SizeIndexEntry: expected 16, got {}",
                bytes.len()
            )
            .into());
        }
        let size = u64::MAX - u64::from_be_bytes(bytes[..8].try_into().unwrap());
        let obj_id = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        Ok(SizeIndexEntry { size, obj_id })
    }
}
#[derive(Debug, PartialEq, Clone)]
#[pyclass(frozen, skip_from_py_object, get_all)]
/// Aggregate counts and sizes for one type.
pub struct TypeSummary {
    pub count: u64,
    pub total_size: u64,
}

impl BytesEncode<'_> for TypeSummary {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&(item.count).to_ne_bytes());
        buf[8..].copy_from_slice(&(item.total_size).to_ne_bytes());
        Ok(buf.to_vec().into())
    }
}

impl BytesDecode<'_> for TypeSummary {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        if bytes.len() != 16 {
            return Err(format!(
                "Invalid byte length for TypeSummary: expected 16, got {}",
                bytes.len()
            )
            .into());
        }
        let count = u64::from_ne_bytes(bytes[..8].try_into()?);
        let total_size = u64::from_ne_bytes(bytes[8..16].try_into()?);
        Ok(TypeSummary { count, total_size })
    }
}

const PRIMARY_DB: &str = "primary";
const REFERRERS_DB: &str = "referrers";
const TYPES_DB: &str = "types";
const TYPES_SIZE_INDEX_DB: &str = "types_size_index";
const TYPES_SUBTREE_SIZE_INDEX_DB: &str = "types_subtree_size_index";
const TYPE_SUMMARIES_DB: &str = "type_summaries";
const ALL_TYPES: &str = "All Types";
const PAGE_SIZE: usize = 1000; // Hardcode this for now

type Id = u64;
type IdDbType = U64<NativeEndian>;

trait SizeEstimator: Clone {
    fn empty() -> Self;
    fn add_in_place(&mut self, element: u64, value: u64);
    fn total(&self) -> u64;
    fn include(&mut self, other: &Self);
}

impl<const N: usize> SizeEstimator for SizeSketch<N> {
    fn empty() -> Self {
        SizeSketch::new()
    }

    fn add_in_place(&mut self, element: u64, value: u64) {
        self.add(element, value as f64);
    }

    fn total(&self) -> u64 {
        self.estimate() as u64
    }

    fn include(&mut self, other: &Self) {
        self.update_in_place(other);
    }
}

impl SizeEstimator for Rc<SummedRadixTree> {
    fn empty() -> Self {
        SummedRadixTree::new()
    }

    fn add_in_place(&mut self, element: u64, value: u64) {
        *self = self.add(element as usize, value);
    }

    fn total(&self) -> u64 {
        SummedRadixTree::total(self)
    }

    fn include(&mut self, other: &Self) {
        *self = self.union(other);
    }
}

#[pyclass(frozen, from_py_object)]
#[derive(Debug, Clone)]
/// Controls the speed and accuracy of subtree-size estimation.
pub enum EstimatorPrecision {
    NoEstimates,
    Low,
    Medium,
    High,
    Exact,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]

enum Type {
    AllTypes(),
    TypeName(String),
}

impl Type {
    fn from_str(s: &str) -> Self {
        if s == ALL_TYPES {
            Type::AllTypes()
        } else {
            Type::TypeName(s.to_string())
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Type {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Type::from_str(obj.extract()?))
    }
}

impl<'py> IntoPyObject<'py> for Type {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let s = match self {
            Type::AllTypes() => ALL_TYPES.to_string(),
            Type::TypeName(name) => name,
        };
        Ok(PyString::new(py, &s))
    }
}

const ALL_TYPES_KEY: &[u8] = b"All Types";

impl<'a> BytesEncode<'a> for Type {
    type EItem = Self;

    fn bytes_encode(item: &Self) -> Result<Cow<'_, [u8]>, Box<dyn Error + Send + Sync>> {
        match item {
            Type::AllTypes() => Ok(Cow::Borrowed(ALL_TYPES_KEY)),
            Type::TypeName(name) => Ok(Cow::Borrowed(name.as_bytes())),
        }
    }
}

impl<'txn> BytesDecode<'txn> for Type {
    type DItem = Self;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        if bytes == ALL_TYPES_KEY {
            Ok(Type::AllTypes())
        } else {
            Ok(Type::TypeName(String::from_str(str::from_utf8(bytes)?)?))
        }
    }
}

#[pyclass]
/// LMDB-backed explorer for heap dumps produced by the Python dumper.
pub struct HeapDumpExplorer {
    env: Env<WithoutTls>,
    primary_db: Database<IdDbType, SerdeJson<RawObjectRecord>>,
    referrers_db: Database<IdDbType, IdDbType>,
    types_db: Database<Type, IdDbType>,
    types_size_index_db: Database<Type, SizeIndexEntry>,
    types_subtree_size_index_db: Database<Type, SizeIndexEntry>,
    type_summaries_db: Database<Type, TypeSummary>,
}

impl HeapDumpExplorer {
    fn just_import_lines(&self, lines: &Bound<'_, PyAny>) -> anyhow::Result<()> {
        let mut tx = self.env.write_txn()?;
        for item in lines.try_iter()? {
            let item = item?;
            let line_result: PyResult<_> = item.extract::<&[u8]>().map_err(|e| e.into());
            let line = line_result?;
            let record: RawObjectRecord = serde_json::from_slice(line)?;
            self.primary_db.put(&mut tx, &record.id, &record)?;

            for reference in &record.references {
                self.referrers_db.put(&mut tx, reference, &record.id)?;
            }

            let type_key = Type::TypeName(record.r#type.clone());
            self.types_db.put(&mut tx, &type_key, &record.id)?;
            self.types_db.put(&mut tx, &Type::AllTypes(), &record.id)?;
            self.put_size_index_entry(
                &mut tx,
                &type_key,
                &SizeIndexEntry {
                    size: record.size,
                    obj_id: record.id,
                },
                self.types_size_index_db,
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn put_size_index_entry(
        &self,
        tx: &mut heed::RwTxn,
        type_key: &Type,
        entry: &SizeIndexEntry,
        size_index: Database<Type, SizeIndexEntry>,
    ) -> anyhow::Result<()> {
        size_index.put(tx, type_key, entry)?;
        size_index.put(tx, &Type::AllTypes(), entry)?;
        Ok(())
    }

    fn run_post_import_processing(
        &self,
        estimator_precision: EstimatorPrecision,
    ) -> anyhow::Result<()> {
        let mut tx = self.env.write_txn()?;
        self.build_type_summaries(&mut tx)?;
        match estimator_precision {
            EstimatorPrecision::NoEstimates => Ok(()), // Don't build any sketches, so we can save time and space if the user doesn't care about estimates.
            EstimatorPrecision::Low => self
                .explore_strongly_connected_components::<'_, '_, LowPrecisionSizeSketch>(&mut tx),
            EstimatorPrecision::Medium => self
                .explore_strongly_connected_components::<'_, '_, MediumPrecisionSizeSketch>(
                    &mut tx,
                ),
            EstimatorPrecision::High => self
                .explore_strongly_connected_components::<'_, '_, HighPrecisionSizeSketch>(&mut tx),
            EstimatorPrecision::Exact => {
                self.explore_strongly_connected_components::<'_, '_, Rc<SummedRadixTree>>(&mut tx)
            }
        }?;
        tx.commit()?;
        Ok(())
    }

    fn build_type_summaries(&self, tx: &mut heed::RwTxn) -> anyhow::Result<()> {
        let rtx = self.env.read_txn()?;
        let mut last_type_key: Option<Type> = None;
        let mut total_size = 0;
        let mut count = 0;
        for type_size_info in self.types_size_index_db.iter(&rtx)? {
            let (type_key, size_index_entry) = type_size_info?;
            if let Some(ref last_key) = last_type_key
                && &type_key != last_key
            {
                self.type_summaries_db
                    .put(tx, last_key, &TypeSummary { count, total_size })?;
                total_size = 0;
                count = 0;
            }
            last_type_key = Some(type_key);
            total_size += size_index_entry.size;
            count += 1;
        }
        if let Some(ref last_key) = last_type_key {
            self.type_summaries_db
                .put(tx, last_key, &TypeSummary { count, total_size })?;
        }
        Ok(())
    }

    fn explore_strongly_connected_components<'vis, 'env, T: SizeEstimator>(
        &'env self,
        tx: &'vis mut heed::RwTxn<'env>,
    ) -> anyhow::Result<()>
    where
        'env: 'vis,
    {
        let ro_txn = self.env.read_txn()?;
        let ro_iter = self
            .primary_db
            .iter(&ro_txn)?
            .remap_data_type::<SerdeJson<RawObjectRecord>>()
            .lazily_decode_data();
        let known_skips = std::collections::HashSet::new();
        let mut visitor = StronglyConnectedComponentsVisitor {
            explorer: self,
            ro_txn: &ro_txn,
            ro_iter,
            rw_txn: tx,
            known_skips,
            _estimator: PhantomData::<T>,
        };
        tarjan::visit_sccs(&mut visitor)?;
        Ok(())
    }

    fn get_summaries_for_ids(
        &self,
        tx: &heed::RoTxn,
        ids: &[Id],
    ) -> anyhow::Result<Vec<ObjectSummary>> {
        let summaries = collect_results(ids.iter().map(|id| self.get_summary(tx, *id)))?;
        Ok(summaries)
    }

    fn retrieve_page_of_object_summaries<T>(
        &self,
        tx: &heed::RoTxn,
        iter: impl Iterator<Item = heed::Result<(T, Id)>>,
        page: Option<usize>,
    ) -> anyhow::Result<Vec<ObjectSummary>> {
        let ids: Vec<heed::Result<(T, Id)>> = if let Some(page) = page {
            iter.skip(page * PAGE_SIZE).take(PAGE_SIZE).collect()
        } else {
            iter.collect()
        };
        let mut result = Vec::with_capacity(ids.len());
        for id_res in ids {
            let (_, id) = id_res?;
            let summary = self.get_summary(tx, id)?;
            result.push(summary);
        }
        Ok(result)
    }

    fn get_summary(&self, tx: &heed::RoTxn, id: Id) -> anyhow::Result<ObjectSummary> {
        self.get_and_decode_record(tx, id)
    }

    fn get_and_decode_record<'a, T: Deserialize<'a> + 'static>(
        &self,
        tx: &'a heed::RoTxn,
        id: Id,
    ) -> anyhow::Result<T> {
        let remapped_db = self.primary_db.remap_data_type::<SerdeJson<T>>();
        remapped_db
            .get(tx, &id)?
            .ok_or_else(|| anyhow::anyhow!("Decoding error: Missing record for ID {id}"))
    }
}

#[pymethods]
impl HeapDumpExplorer {
    #[new]
    /// Open or create the LMDB environment at `db_path`.
    fn new(db_path: String) -> anyhow::Result<Self> {
        let env = unsafe {
            EnvOpenOptions::new()
                .read_txn_without_tls()
                .map_size(64 * 1024 * 1024 * 1024)
                .max_dbs(7)
                .open(db_path)
        }?;
        let mut wtxn = env.write_txn()?;
        #[allow(deprecated)]
        // Deprecation warning says to use IntegerComparator, but it's too slow to put in hot loops
        let primary_db = env
            .database_options()
            .name(PRIMARY_DB)
            .flags(DatabaseFlags::INTEGER_KEY)
            .types()
            .create(&mut wtxn)?;
        #[allow(deprecated)]
        // Deprecation warning says to use IntegerComparator, but it's too slow to put in hot loops
        let referrers_db = env
            .database_options()
            .name(REFERRERS_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED | DatabaseFlags::INTEGER_KEY)
            .types()
            .create(&mut wtxn)?;
        let types_db = env
            .database_options()
            .name(TYPES_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .create(&mut wtxn)?;
        let types_size_index_db = env
            .database_options()
            .name(TYPES_SIZE_INDEX_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .create(&mut wtxn)?;
        let types_subtree_size_index_db = env
            .database_options()
            .name(TYPES_SUBTREE_SIZE_INDEX_DB)
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .types()
            .create(&mut wtxn)?;
        let type_summaries_db = env
            .database_options()
            .name(TYPE_SUMMARIES_DB)
            .types()
            .create(&mut wtxn)?;
        wtxn.commit()?;

        Ok(Self {
            env,
            primary_db,
            referrers_db,
            types_db,
            types_size_index_db,
            types_subtree_size_index_db,
            type_summaries_db,
        })
    }

    /// Import JSONL heap records and build the requested secondary indexes.
    fn import_lines(
        &self,
        lines: &Bound<'_, PyAny>,
        estimator_precision: EstimatorPrecision,
    ) -> anyhow::Result<()> {
        self.just_import_lines(lines)?;
        lines
            .py()
            .detach(|| self.run_post_import_processing(estimator_precision))?;
        Ok(())
    }

    /// Return one object with expanded references and referrers.
    fn get_object(&self, obj_id: Id) -> anyhow::Result<Option<ObjectRecord>> {
        let rtxn = self.env.read_txn()?;
        if let Some(record) = self.primary_db.get(&rtxn, &obj_id)? {
            let referrers_iter_opt = self.referrers_db.get_duplicates(&rtxn, &obj_id)?;

            let referrers = match referrers_iter_opt {
                Some(iter) => self.retrieve_page_of_object_summaries(&rtxn, iter, None)?,
                None => vec![],
            };

            Ok(Some(ObjectRecord {
                id: record.id,
                r#type: record.r#type,
                value: record.value,
                size: record.size,
                subtree_size: record.subtree_size,
                references: self.get_summaries_for_ids(&rtxn, &record.references)?,
                referrers,
            }))
        } else {
            Ok(None)
        }
    }

    /// Return summary rows for every known type.
    fn get_type_summaries(&self) -> anyhow::Result<Vec<(Type, TypeSummary)>> {
        let rtxn = self.env.read_txn()?;
        let summaries = collect_results(self.type_summaries_db.iter(&rtxn)?)?;
        Ok(summaries)
    }

    /// Return the number of objects recorded for one type.
    fn get_count_for_type(&self, typename: Type) -> anyhow::Result<usize> {
        let rtxn = self.env.read_txn()?;
        Ok(self
            .type_summaries_db
            .get(&rtxn, &typename)?
            .map(|summary| summary.count)
            .unwrap_or(0) as usize)
    }

    /// Return the number of result pages for one type.
    fn get_page_count_for_type(&self, typename: Type) -> anyhow::Result<usize> {
        let count = self.get_count_for_type(typename)?;
        Ok(count.div_ceil(PAGE_SIZE))
    }

    /// Return one page of objects for a type in id order.
    fn get_objects_by_type(
        &self,
        typename: Type,
        page: Option<usize>,
    ) -> anyhow::Result<Vec<ObjectSummary>> {
        let rtxn = self.env.read_txn()?;
        if let Some(iter) = self.types_db.get_duplicates(&rtxn, &typename)? {
            self.retrieve_page_of_object_summaries(&rtxn, iter, page)
        } else {
            Ok(vec![])
        }
    }

    /// Return one page of objects ordered by direct size or estimated subtree size.
    fn get_objects_by_type_ordered_by_size(
        &self,
        r#type: Type,
        subtree_size: bool,
        page: Option<usize>,
    ) -> anyhow::Result<Vec<ObjectSummary>> {
        let rtxn = self.env.read_txn()?;
        let index_db = if subtree_size {
            &self.types_subtree_size_index_db
        } else {
            &self.types_size_index_db
        };
        if let Some(iter) = index_db.get_duplicates(&rtxn, &r#type)? {
            self.retrieve_page_of_object_summaries(
                &rtxn,
                iter.map(|res| res.map(|(_, index_entry)| (index_entry.size, index_entry.obj_id))),
                page,
            )
        } else {
            Ok(vec![])
        }
    }

    /// Find one reference path between two objects, optionally skipping specific IDs.
    fn find_path_between_objects(
        &self,
        start_id: Id,
        end_id: Id,
        avoiding_ids: HashSet<Id>,
    ) -> anyhow::Result<Option<Vec<ObjectSummary>>> {
        let rtxn = self.env.read_txn()?;
        let mut queue: VecDeque<RawObjectRecord> = VecDeque::new();
        queue.push_back(self.get_and_decode_record(&rtxn, start_id)?);
        let endpoint: RawObjectRecord = self.get_and_decode_record(&rtxn, end_id)?;
        let endpoint_subtree_size = endpoint.subtree_size;

        let mut predecessors = HashMap::new();
        predecessors.insert(start_id, None); // Also acts as the visited set.
        let mut dead_ends = HashSet::new();
        while let Some(current_obj) = queue.pop_front() {
            if let (Some(current_subtree_size), Some(endpoint_subtree_size)) =
                (current_obj.subtree_size, endpoint_subtree_size)
            {
                // Smaller subtrees cannot contain the endpoint, so skip them early.
                // Whilst you might worry that this would be incorrect for approximate subtree sizes,
                // the sketches we use are monotonic, so a smaller estimated subtree size does
                // indeed guarantee a smaller true subtree size.
                if current_subtree_size < endpoint_subtree_size {
                    dead_ends.insert(current_obj.id);
                    continue;
                }
            }
            if current_obj.id == end_id {
                // Walk the predecessor chain back to the start.
                let mut path = Vec::new();
                let mut current_opt = Some(current_obj.id);
                while let Some(current_id) = current_opt {
                    let summary = self.get_summary(&rtxn, current_id)?;
                    path.push(summary);
                    current_opt = predecessors[&current_id];
                }
                return Ok(Some(path.into_iter().rev().collect()));
            }
            for ref_id in current_obj.references {
                if dead_ends.contains(&ref_id) {
                    continue; // Skip branches we already proved cannot succeed.
                }
                if avoiding_ids.contains(&ref_id) {
                    continue;
                }
                let obj = self.get_and_decode_record(&rtxn, ref_id)?;
                if should_skip_link_in_subtree_exploration(&obj) {
                    dead_ends.insert(ref_id);
                    continue;
                }

                if let std::collections::hash_map::Entry::Vacant(e) = predecessors.entry(ref_id) {
                    e.insert(Some(current_obj.id));
                    queue.push_back(obj);
                }
            }
        }
        Ok(None)
    }

    /// Find the largest object reachable from both of a pair of objects
    fn find_largest_common_reachable_object(
        &self,
        obj_id1: Id,
        obj_id2: Id,
    ) -> anyhow::Result<Option<Id>> {
        let rtxn = self.env.read_txn()?;

        #[derive(Eq, PartialEq)]
        enum Obj1Or2 {
            Obj1,
            Obj2,
        }
        #[derive(Eq, PartialEq)]
        struct QueueItem {
            id: Id,
            subtree_size: u64,
            successors: Vec<Id>,
            obj_1_or_2: Obj1Or2,
        }

        impl Ord for QueueItem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.subtree_size.cmp(&other.subtree_size)
            }
        }

        impl PartialOrd for QueueItem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut queue = BinaryHeap::<QueueItem>::new();
        let mut visited_from_1 = HashSet::<Id>::new();
        let mut visited_from_2 = HashSet::<Id>::new();

        fn make_queue_item(
            record: &RawObjectRecord,
            obj_1_or_2: Obj1Or2,
        ) -> anyhow::Result<QueueItem> {
            Ok(QueueItem {
                id: record.id,
                subtree_size: record.expect_subtree_size()?,
                successors: record.references.clone(),
                obj_1_or_2,
            })
        }

        queue.push(make_queue_item(
            &self.get_and_decode_record(&rtxn, obj_id1)?,
            Obj1Or2::Obj1,
        )?);
        queue.push(make_queue_item(
            &self.get_and_decode_record(&rtxn, obj_id2)?,
            Obj1Or2::Obj2,
        )?);

        while let Some(next) = queue.pop() {
            match next.obj_1_or_2 {
                Obj1Or2::Obj1 => {
                    if !visited_from_1.insert(next.id) {
                        continue;
                    }
                    if visited_from_2.contains(&next.id) {
                        return Ok(Some(next.id));
                    }
                    for succ_id in next.successors {
                        let record = self.get_and_decode_record(&rtxn, succ_id)?;
                        if !should_skip_link_in_subtree_exploration(&record) {
                            queue.push(make_queue_item(&record, Obj1Or2::Obj1)?);
                        }
                    }
                }
                Obj1Or2::Obj2 => {
                    if !visited_from_2.insert(next.id) {
                        continue;
                    }
                    if visited_from_1.contains(&next.id) {
                        return Ok(Some(next.id));
                    }
                    for succ_id in next.successors {
                        let record = self.get_and_decode_record(&rtxn, succ_id)?;
                        if !should_skip_link_in_subtree_exploration(&record) {
                            queue.push(make_queue_item(&record, Obj1Or2::Obj2)?);
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Find the largest descendant of an object that has a smaller subtree.
    fn find_largest_strict_descendant(&self, obj_id: Id) -> anyhow::Result<Option<Id>> {
        let rtxn = self.env.read_txn()?;
        let mut queue: VecDeque<Id> = VecDeque::new();
        let start_record: RawObjectRecord = self.get_and_decode_record(&rtxn, obj_id)?;
        let start_subtree_size = start_record.expect_subtree_size()?;
        let mut largest_id: Option<Id> = None;
        let mut largest_subtree = 0;
        let mut visited = HashSet::new();
        visited.insert(obj_id);
        for succ_id in start_record.references {
            queue.push_back(succ_id);
        }
        while let Some(next) = queue.pop_front() {
            let record: RawObjectRecord = self.get_and_decode_record(&rtxn, next)?;
            if should_skip_link_in_subtree_exploration(&record) || visited.contains(&next) {
                continue;
            }
            visited.insert(next);
            let next_subtree_size = record.expect_subtree_size()?;
            if next_subtree_size < start_subtree_size && next_subtree_size > largest_subtree {
                largest_id = Some(next);
                largest_subtree = next_subtree_size;
            } else {
                for succ_id in record.references {
                    queue.push_back(succ_id);
                }
            }
        }
        return Ok(largest_id);
    }

    /// Find the largest ancestor of an object
    fn find_largest_ancestor(&self, obj_id: Id) -> anyhow::Result<Id> {
        let rtxn = self.env.read_txn()?;
        let mut queue: VecDeque<Id> = VecDeque::new();
        let mut largest_id: Id = obj_id;
        let mut largest_subtree = 0;
        let mut visited = HashSet::new();
        queue.push_back(obj_id);
        while let Some(next) = queue.pop_front() {
            let record: RawObjectRecord = self.get_and_decode_record(&rtxn, next)?;
            if visited.contains(&next) {
                continue;
            }
            visited.insert(next);
            let next_subtree_size = record.expect_subtree_size()?;
            if next_subtree_size > largest_subtree {
                largest_id = next;
                largest_subtree = next_subtree_size;
            }
            /* For exploring ancestors, we do the skip check _after_ considering the current node.
            The skip checks assume you're exploring descendants. If we're exploring ancestors,
            we don't consider ancestors of skipped nodes. */
            if should_skip_link_in_subtree_exploration(&record) {
                continue;
            }
            if let Some(referrers_iter) = self.referrers_db.get_duplicates(&rtxn, &next)? {
                for x in referrers_iter {
                    let (_, referrer_id) = x?;
                    queue.push_back(referrer_id);
                }
            }
        }
        Ok(largest_id)
    }
}

struct StronglyConnectedComponentsVisitor<'vis, 'env, SizeEstimatorT: SizeEstimator>
where
    'env: 'vis,
{
    explorer: &'vis HeapDumpExplorer,
    ro_txn: &'vis heed::RoTxn<'env>,
    ro_iter: heed::RoIter<'vis, IdDbType, LazyDecode<SerdeJson<RawObjectRecord>>>,
    rw_txn: &'vis mut heed::RwTxn<'env>,
    known_skips: std::collections::HashSet<Id>,
    _estimator: PhantomData<SizeEstimatorT>,
}

fn should_skip_link_in_subtree_exploration(record: &RawObjectRecord) -> bool {
    /* Skip:
        - modules, since it's not useful to include them in subtree exploration.
        - types, so that instance sizes are not inflated by including type objects.
        - functions from various stdlib modules that bring in big subtrees
    */
    regex!(r"^(builtins.(module|type)$|(typing|copyreg|copy|contextlib|abc|ast)\.)")
        .is_match(&record.r#type)
        || record.r#type == "builtins.function"
            && record
                .value
                .as_deref()
                .is_some_and(|x| regex!(r"^(typing|copyreg|copy|contextlib|abc|ast)\.").is_match(x))
}

impl<'vis, 'env, T: SizeEstimator> GraphSCCVisitor
    for StronglyConnectedComponentsVisitor<'vis, 'env, T>
{
    type NodeT = RawObjectRecord;
    type NodeIdT = Id;
    type NodeAccT = usize;
    type SCCAccT = T;
    type ErrorT = anyhow::Error;

    fn next_unvisited_node(
        &mut self,
        mut already_visited: impl FnMut(&Self::NodeIdT) -> bool,
    ) -> Result<Option<Self::NodeT>, Self::ErrorT> {
        for item in &mut self.ro_iter {
            let (node_id, record): (Self::NodeIdT, Lazy<SerdeJson<RawObjectRecord>>) = item?;
            if !already_visited(&node_id) {
                let record = record.decode().map_err(HeedError::Decoding)?;
                return Ok(Some(record));
            }
        }
        Ok(None)
    }

    fn get_node_id(&self, node: &Self::NodeT) -> Self::NodeIdT {
        node.id
    }

    fn get_node_acc(&self, node: &Self::NodeT) -> Self::NodeAccT {
        node.size as usize
    }

    fn get_successors(&mut self, node: &Self::NodeT) -> Result<Vec<Self::NodeT>, Self::ErrorT> {
        let remapped_db = self
            .explorer
            .primary_db
            .remap_data_type::<SerdeJson<RawObjectRecord>>();
        let results = collect_results(node.references.iter().filter_map(|succ_id| {
            if self.known_skips.contains(succ_id) {
                return None;
            }

            match remapped_db.get(self.ro_txn, succ_id) {
                Ok(Some(record)) => {
                    if should_skip_link_in_subtree_exploration(&record) {
                        self.known_skips.insert(*succ_id);
                        None
                    } else {
                        Some(Ok(record))
                    }
                }
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        }))?;
        Ok(results)
    }

    fn accumulate_node_values(&self, v1: &mut Self::NodeAccT, v2: &Self::NodeAccT) {
        *v1 += *v2;
    }

    fn accumulate_scc_values(&self, v1: &mut Self::SCCAccT, v2: &Self::SCCAccT) {
        v1.include(v2);
    }

    fn add_node_value_to_scc_value(
        &self,
        node_acc: &Self::NodeAccT,
        this_scc: usize,
        scc_acc: Option<&Self::SCCAccT>,
    ) -> Self::SCCAccT {
        let mut scc_acc = scc_acc.cloned().unwrap_or_else(|| T::empty());
        scc_acc.add_in_place(this_scc as u64, *node_acc as u64);
        scc_acc
    }

    fn emit_result(
        &mut self,
        node_id: &Self::NodeIdT,
        _node_acc_this_scc: &Self::NodeAccT,
        scc_acc: &Self::SCCAccT,
    ) -> Result<(), Self::ErrorT> {
        let mut record: RawObjectRecord =
            self.explorer.get_and_decode_record(self.rw_txn, *node_id)?;
        let subtree_size = scc_acc.total();
        record.subtree_size = Some(subtree_size);
        self.explorer
            .primary_db
            .put(self.rw_txn, node_id, &record)?;
        self.explorer.put_size_index_entry(
            self.rw_txn,
            &Type::TypeName(record.r#type),
            &SizeIndexEntry {
                size: subtree_size,
                obj_id: *node_id,
            },
            self.explorer.types_subtree_size_index_db,
        )?;
        Ok(())
    }
}

fn collect_results<T, E>(iter: impl IntoIterator<Item = Result<T, E>>) -> Result<Vec<T>, E> {
    iter.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_summary_deserialization() {
        let json = r#"{"id": 1, "type": "int", "value": "42", "size": 28, "subtree_size": 100, "references": []}"#;
        let summary: ObjectSummary = serde_json::from_str(json).unwrap();
        assert_eq!(
            summary,
            ObjectSummary {
                id: 1,
                r#type: "int".to_string(),
                value: Some("42".to_string()),
                size: 28,
                subtree_size: Some(100),
            }
        );
    }

    #[test]
    fn test_object_summary_deserialization_opt_fields_empty() {
        let json = r#"{"id": 1, "type": "int", "size": 28}"#;
        let summary: ObjectSummary = serde_json::from_str(json).unwrap();
        assert_eq!(
            summary,
            ObjectSummary {
                id: 1,
                r#type: "int".to_string(),
                value: None,
                size: 28,
                subtree_size: None,
            }
        );
    }
}
