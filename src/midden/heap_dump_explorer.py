"""A class that allows exploring heap dumps exported by dump_heap.py.
It uses LMDB to store the data on disk and provides methods for querying objects, their types, and their relationships.
"""

from midden.tarjan import GraphSCCVisitor, visit_sccs

from midden.set_sketch import SetSketch

from dataclasses import dataclass

import struct
from collections import Counter, deque
from lmdb import Environment, Transaction, _Database
from threading import local
from functools import wraps
from typing import (
    ParamSpec,
    Callable,
    TypeVar,
    Iterable,
)

from pydantic import BaseModel, ConfigDict


P = ParamSpec("P")
T = TypeVar("T")
F = Callable[P, T]
M = TypeVar("M", bound=BaseModel)


class ObjectSummary(BaseModel):
    id: int
    type: str
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0
    model_config = ConfigDict(extra="ignore")


class _RawObjectRecord(BaseModel):
    id: int
    type: str
    references: list[int]
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0
    model_config = ConfigDict(extra="forbid")


class ObjectRecord(BaseModel):
    id: int
    type: str
    references: list[ObjectSummary]
    referrers: list[ObjectSummary]
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0


class _ObjectRecordNoValue(BaseModel):
    id: int
    type: str
    references: list[int]
    size: int
    model_config = ConfigDict(extra="ignore")


class _SizeIndexEntry(BaseModel):
    size: int
    obj_id: int

    def _pack(self) -> bytes:
        # Pack by size descending, then by obj_id ascending, so we can sort by size in LMDB
        return struct.pack(">qQ", -self.size, self.obj_id)

    @staticmethod
    def _unpack(data: bytes) -> "_SizeIndexEntry":
        size, obj_id = struct.unpack(">qQ", data)
        return _SizeIndexEntry(size=-size, obj_id=obj_id)


PRIMARY_DB = b"primary"
REFERRERS_DB = b"referrers"
TYPES_DB = b"types"
TYPES_SIZE_INDEX_DB = b"types_size_index"
TYPES_SUBTREE_SIZE_INDEX_DB = b"types_subtree_size_index"
SCCS_SKETCH_DB = b"sccs_sketch"
PAGE_SIZE = 1000  # Hardcode this for now


def _pack_id(obj_id: int) -> bytes:
    return struct.pack("@n", obj_id)


def _unpack_id(data: bytes) -> int:
    return struct.unpack("@n", data)[0]


_tx_local: local[Transaction] = local()


def tx(f: F) -> F:
    return _wrap_txn(f, write=False)


def tx_write(f: F) -> F:
    return _wrap_txn(f, write=True)


def _wrap_txn(f: F, write=False) -> F:
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self = args[0]
        assert isinstance(self, HeapDumpExplorer)
        if hasattr(_tx_local, "txn"):
            # Support nested transactions by reusing the same one. LMDB supports this.
            return f(*args, **kwargs)
        with self._env.begin(write=write) as txn:
            _tx_local.txn = txn
            try:
                return f(*args, **kwargs)
            finally:
                del _tx_local.txn

    return wrapper


def txn() -> Transaction:
    return _tx_local.txn


class HeapDumpExplorer:
    def __init__(self, db_path="/tmp/heap_dump_db"):
        self._env = Environment(
            db_path, map_size=10 * 1024 * 1024 * 1024, max_dbs=6
        )  # 10 GB
        self._primary_db = self._env.open_db(PRIMARY_DB, integerkey=True, create=True)
        # The referrers_db is a reverse index of the primary_db, mapping
        # from an object ID to the IDs of objects that reference it, so
        # we can efficiently look up referrers for any object.
        self._referrers_db = self._env.open_db(
            REFERRERS_DB, integerdup=True, create=True
        )
        # The types_db maps from type name to the IDs of objects of that type, so we can efficiently look up all objects of a given type.
        self._types_db = self._env.open_db(TYPES_DB, dupsort=True, create=True)
        # The types_size_index_db and types_subtree_size_index_db are indexes that allow us to look up objects of a given type
        # ordered by size or subtree size, respectively, for efficient pagination of large types.
        self._types_size_index_db = self._env.open_db(
            TYPES_SIZE_INDEX_DB, dupsort=True, create=True
        )
        self._types_subtree_size_index_db = self._env.open_db(
            TYPES_SUBTREE_SIZE_INDEX_DB, dupsort=True, create=True
        )
        # The sccs_sketch_db stores a sketch of which SCCs are reachable from each object, to allow for efficient path finding
        # queries that can quickly rule out objects that can't possibly reach the target.
        self._sccs_sketch_db = self._env.open_db(
            SCCS_SKETCH_DB, integerkey=True, create=True
        )

    def import_dump(self, dump_path="/tmp/dump.jsonl"):
        with open(dump_path, "rb") as f:
            self.import_lines(f)

    @tx_write
    def import_lines(self, lines: Iterable[bytes]):
        for line in lines:
            record = _RawObjectRecord.model_validate_json(line)
            obj_id = record.id
            encoded_obj_id = _pack_id(obj_id)

            txn().put(encoded_obj_id, line, db=self._primary_db)

            for reference in record.references:
                txn().put(
                    _pack_id(reference),
                    encoded_obj_id,
                    dupdata=True,
                    db=self._referrers_db,
                )

            type_name = record.type
            txn().put(
                type_name.encode(), encoded_obj_id, dupdata=True, db=self._types_db
            )

            self._put_size_index_entry(
                obj_id, type_name, record.size, self._types_size_index_db
            )

        self._explore_strongly_connected_components()

    @tx
    def get_object(self, obj_id: int) -> ObjectRecord | None:
        data = self._load_and_validate(obj_id, _RawObjectRecord)
        if data is None:
            return None

        referrers = []
        cursor = txn().cursor(db=self._referrers_db)
        if cursor.set_key(_pack_id(obj_id)):
            for value in cursor.iternext_dup():
                referrers.append(_unpack_id(value))

        return ObjectRecord(
            id=data.id,
            type=data.type,
            value=data.value,
            size=data.size,
            subtree_size=data.subtree_size,
            references=self._get_summaries_for_ids(data.references),
            referrers=self._get_summaries_for_ids(referrers),
        )

    def _get_summaries_for_ids(self, ids: list[int]) -> list[ObjectSummary]:
        summaries = []
        for obj_id in ids:
            summary = self._load_and_validate(obj_id, ObjectSummary)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _load_and_validate(self, id: int, model: type[M]) -> M | None:
        data = txn().get(_pack_id(id), db=self._primary_db)
        if data is None:
            return None
        return model.model_validate_json(data)

    @tx
    def get_type_counts(self) -> list[tuple[str, int]]:
        type_counts: Counter[str] = Counter()
        cursor = txn().cursor(db=self._types_db)
        for key in cursor.iternext_nodup(keys=True, values=False):
            assert isinstance(key, bytes)
            type_counts[key.decode()] += cursor.count()
        return type_counts.most_common()

    @tx
    def get_count_for_type(self, type_name: str) -> int:
        cursor = txn().cursor(db=self._types_db)
        if cursor.set_key(type_name.encode()):
            return cursor.count()
        return 0

    @tx
    def get_page_count_for_type(self, type_name: str) -> int:
        count = self.get_count_for_type(type_name)
        return (count + PAGE_SIZE - 1) // PAGE_SIZE

    @tx
    def get_objects_by_type(
        self, type_name: str, page: int | None = None
    ) -> list[ObjectSummary]:
        objects: list[ObjectSummary] = []
        cursor = txn().cursor(db=self._types_db)
        if cursor.set_key(type_name.encode()):
            if page is not None:
                for _ in range(page * PAGE_SIZE):
                    # If this naive pagination proves too slow, we can build a separate index for pagination
                    if not cursor.next_dup():
                        return []
            for i, value in enumerate(cursor.iternext_dup()):
                if page is not None and i >= PAGE_SIZE:
                    break
                obj_id = _unpack_id(value)
                summary = self._load_and_validate(obj_id, ObjectSummary)
                if summary is not None:
                    objects.append(summary)

        return objects

    @tx
    def get_objects_by_type_ordered_by_size(
        self, type_name: str, subtree_size=False, page: int | None = None
    ) -> list[ObjectSummary]:
        objects: list[ObjectSummary] = []
        index_db = (
            self._types_subtree_size_index_db
            if subtree_size
            else self._types_size_index_db
        )
        cursor = txn().cursor(db=index_db)
        if cursor.set_key(type_name.encode()):
            if page is not None:
                for _ in range(page * PAGE_SIZE):
                    if not cursor.next_dup():
                        return []
            for i, value in enumerate(cursor.iternext_dup()):
                if page is not None and i >= PAGE_SIZE:
                    break
                index_entry = _SizeIndexEntry._unpack(value)
                obj_id = index_entry.obj_id
                summary = self._load_and_validate(obj_id, ObjectSummary)
                if summary is not None:
                    objects.append(summary)

        return objects

    @tx
    def find_path_between_objects(
        self, start_id: int, end_id: int, avoid_ids: set[int] | None = None
    ) -> list[ObjectSummary] | None:

        queue = deque([start_id])
        predecessors = {start_id: None}  # Doubles as a visited set
        dead_ends = set()
        start_sketch = self._get_scc_sketch(start_id)
        end_sketch = self._get_scc_sketch(end_id)
        if not end_sketch.is_subset_of(start_sketch):
            return None  # No path can exist if end's reachable SCCs aren't a subset of start's reachable SCCs
        while queue:
            current_id = queue.popleft()
            if current_id == end_id:
                # Reconstruct path
                path = []
                while current_id is not None:
                    summary = self._load_and_validate(current_id, ObjectSummary)
                    assert summary is not None
                    path.append(summary)
                    current_id = predecessors[current_id]
                return list(reversed(path))

            if current_id in dead_ends:
                continue  # Skip known dead ends
            current_sketch = self._get_scc_sketch(current_id)
            if not end_sketch.is_subset_of(current_sketch):
                dead_ends.add(current_id)
                continue  # No path can exist from current to end, so skip it

            object_record = self._load_and_validate(current_id, _ObjectRecordNoValue)
            if self._should_skip_link_in_subtree_exploration(object_record):
                dead_ends.add(current_id)
                continue
            assert object_record is not None
            for ref_id in object_record.references:
                if avoid_ids is not None and ref_id in avoid_ids:
                    continue
                if ref_id not in predecessors:
                    predecessors[ref_id] = current_id
                    queue.append(ref_id)

    def _put_size_index_entry(
        self, obj_id: int, type_name: str, size: int, db: _Database
    ):
        size_index_entry = _SizeIndexEntry(size=size, obj_id=obj_id)
        txn().put(
            type_name.encode(),
            size_index_entry._pack(),
            dupdata=True,
            db=db,
        )

    def _put_sccs_sketch(self, obj_id: int, sccs: SetSketch):
        txn().put(
            _pack_id(obj_id),
            sccs.to_bytes(),
            db=self._sccs_sketch_db,
        )

    def _get_scc_sketch(self, obj_id: int) -> SetSketch:
        data = txn().get(_pack_id(obj_id), db=self._sccs_sketch_db)
        assert data is not None, f"No sketch found for obj_id {obj_id}"
        return SetSketch(from_bytes=data)

    def _should_skip_link_in_subtree_exploration(
        self, link: _ObjectRecordNoValue | None
    ) -> bool:
        # Don't include references from modules in the graph, since they create huge SCCs that aren't interesting
        return link is None or link.type == "builtins.module"

    def _explore_strongly_connected_components(outer_self):
        """Run Tarjan's algorithm to find strongly connected components (SCCs) in the object graph.

        This is used to calculate the subtree sizes for each object, as well as to build sketches
        of which SCCs are reachable from each object for efficient path finding queries."""
        t = txn()

        @dataclass(slots=True)
        class WalkResult:
            subtree_size: int
            size: int
            scc_sketch: SetSketch

        class ObjectGraphVisitor(
            GraphSCCVisitor[_ObjectRecordNoValue, int, int, WalkResult]
        ):
            def __init__(self):
                super().__init__()
                self.known_skips: set[int] = set()

            def accumulate_node_values(self, v1: int, v2: int) -> int:
                return v1 + v2

            def accumulate(
                self,
                node_acc: int,
                this_scc: int,
                scc_values: Iterable[tuple[int, WalkResult]],
            ) -> WalkResult:
                subtree_size = node_acc + sum(
                    child_scc.size for _, child_scc in scc_values
                )
                scc_sketch = (
                    SetSketch()
                    .add_all(child_scc_id for child_scc_id, _ in scc_values)
                    .add(this_scc)
                )
                return WalkResult(
                    size=node_acc, subtree_size=subtree_size, scc_sketch=scc_sketch
                )

            def iterate_nodes(
                self, already_visited: Callable[[int], bool]
            ) -> Iterable[_ObjectRecordNoValue]:
                with t.cursor(db=outer_self._primary_db) as cursor:
                    while cursor.next():
                        obj_id = _unpack_id(cursor.key())
                        if already_visited(obj_id):
                            continue
                        record = _ObjectRecordNoValue.model_validate_json(
                            cursor.value()
                        )
                        yield record

            def get_node_id(self, node: _ObjectRecordNoValue) -> int:
                return node.id

            def get_node_acc(self, node: _ObjectRecordNoValue) -> int:
                return node.size

            def get_successors(
                self, node: _ObjectRecordNoValue
            ) -> Iterable[_ObjectRecordNoValue]:
                for ref_id in node.references:
                    if ref_id in self.known_skips:
                        continue
                    ref_data = t.get(_pack_id(ref_id), db=outer_self._primary_db)
                    if ref_data is None:
                        self.known_skips.add(ref_id)
                        continue
                    ref_record = _ObjectRecordNoValue.model_validate_json(ref_data)
                    if outer_self._should_skip_link_in_subtree_exploration(ref_record):
                        self.known_skips.add(ref_id)
                        continue
                    yield ref_record

            def emit_result(self, node_id: int, scc_acc: WalkResult):
                record = outer_self._load_and_validate(node_id, _RawObjectRecord)
                assert record is not None
                record.subtree_size = scc_acc.subtree_size
                t.put(
                    _pack_id(node_id),
                    record.model_dump_json().encode(),
                    db=outer_self._primary_db,
                )
                outer_self._put_size_index_entry(
                    node_id,
                    record.type,
                    scc_acc.subtree_size,
                    outer_self._types_subtree_size_index_db,
                )
                outer_self._put_sccs_sketch(node_id, scc_acc.scc_sketch)

        visit_sccs(ObjectGraphVisitor())
