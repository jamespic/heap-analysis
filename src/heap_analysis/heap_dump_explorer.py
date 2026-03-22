"""A class that allows exploring heap dumps exported by dump_heap.py.
It uses LMDB to store the data on disk and provides methods for querying objects, their types, and their relationships.
"""

from heap_analysis.long_stack import run_with_long_stack
from dataclasses import dataclass

import struct
from collections import Counter
from lmdb import Environment, Transaction
from threading import local
from functools import wraps
from typing import ParamSpec, Callable, TypeVar, Generic, overload, Literal, Iterable

from pydantic import BaseModel, ConfigDict


P = ParamSpec("P")
T = TypeVar("T")
F = Callable[P, T]


class ObjectRecord(BaseModel, Generic[T]):
    id: int
    type: str
    references: list[T]
    referrers: list[T] = []
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0


class ObjectSummary(BaseModel):
    id: int
    type: str
    value: str | int | float | None = None
    size: int
    subtree_size: int = 0
    model_config = ConfigDict(extra="ignore")


PRIMARY_DB = b"primary"
REFERRERS_DB = b"referrers"
TYPES_DB = b"types"


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
            db_path, map_size=10 * 1024 * 1024 * 1024, max_dbs=3
        )  # 10 GB
        self._primary_db = self._env.open_db(PRIMARY_DB, integerkey=True, create=True)
        self._referrers_db = self._env.open_db(
            REFERRERS_DB, integerdup=True, create=True
        )
        self._types_db = self._env.open_db(TYPES_DB, dupsort=True, create=True)

    def import_dump(self, dump_path="/tmp/dump.jsonl"):
        with open(dump_path, "rb") as f:
            self.import_lines(f)

    @tx_write
    def import_lines(self, lines: Iterable[bytes]):
        for line in lines:
            record = ObjectRecord.model_validate_json(line)
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

        self._calculate_subtree_sizes()

    @overload
    def get_object(
        self, obj_id, references: Literal["none"]
    ) -> ObjectRecord[int] | None: ...

    @overload
    def get_object(
        self, obj_id, references: Literal["ids"]
    ) -> ObjectRecord[int] | None: ...

    @overload
    def get_object(
        self, obj_id, references: Literal["summaries"]
    ) -> ObjectRecord[ObjectSummary | None] | None: ...

    @tx
    def get_object(
        self, obj_id, references: Literal["none", "ids", "summaries"]
    ) -> ObjectRecord[int] | ObjectRecord[ObjectSummary | None] | None:
        data = txn().get(_pack_id(obj_id), db=self._primary_db)
        if data is None:
            return None
        data = ObjectRecord[int].model_validate_json(data)
        if references == "none":
            return data

        referrers = []
        cursor = txn().cursor(db=self._referrers_db)
        if cursor.set_key(_pack_id(obj_id)):
            for value in cursor.iternext_dup():
                referrers.append(_unpack_id(value))
        data.referrers = referrers
        if references == "ids":
            return data

        return ObjectRecord[ObjectSummary | None](
            id=data.id,
            type=data.type,
            value=data.value,
            size=data.size,
            references=[self.get_object_summary(ref_id) for ref_id in data.references],
            referrers=[self.get_object_summary(ref_id) for ref_id in data.referrers],
        )

    @tx
    def get_object_summary(self, obj_id) -> ObjectSummary | None:
        data = txn().get(_pack_id(obj_id), db=self._primary_db)
        if data is None:
            return None
        return ObjectSummary.model_validate_json(data)

    @tx
    def get_type_counts(self) -> list[tuple[str, int]]:
        type_counts: Counter[str] = Counter()
        cursor = txn().cursor(db=self._types_db)
        for key in cursor.iternext(keys=True, values=False):
            assert isinstance(key, bytes)
            type_counts[key.decode()] += 1
        return type_counts.most_common()

    @tx
    def get_objects_by_type(self, type_name: str) -> list[ObjectSummary]:
        objects: list[ObjectSummary] = []
        cursor = txn().cursor(db=self._types_db)
        if cursor.set_key(type_name.encode()):
            for value in cursor.iternext_dup():
                obj_id = _unpack_id(value)
                summary = self.get_object_summary(obj_id)
                if summary is not None:
                    objects.append(summary)

        return objects

    @tx
    def _calculate_subtree_sizes(self):
        # Calculate subtree sizes using Tarjan
        t = txn()

        @dataclass(slots=True)
        class StackEntry:
            index: int
            lowlink: int
            on_stack: bool

        bookkeeping: dict[int, StackEntry] = {}
        stack: list[int] = []
        index = 0
        scc_sizes: dict[int, int] = {}
        obj_id_to_scc: dict[int, int] = {}

        def strongconnect(
            obj_id: int,
        ) -> set[int]:  # Returns set of SCCs that are children of this node
            nonlocal index
            entry = StackEntry(index=index, lowlink=index, on_stack=True)
            bookkeeping[obj_id] = entry
            stack.append(obj_id)
            index += 1
            child_sccs: set[int] = set()

            obj = self.get_object(obj_id, references="ids")
            for ref_id in obj.references:
                bookkeeping_entry = bookkeeping.get(ref_id)
                if bookkeeping_entry is None:
                    child_sccs.update((yield strongconnect(ref_id)))
                    entry.lowlink = min(entry.lowlink, bookkeeping[ref_id].lowlink)
                elif bookkeeping_entry.on_stack:
                    # Use lowlink variant, so lowlink will point to the root of the SCC, not just the first node we saw
                    entry.lowlink = min(entry.lowlink, bookkeeping_entry.index)
                else:
                    child_sccs.add(obj_id_to_scc[ref_id])

            if entry.index == entry.lowlink:
                # Found an SCC root, pop the stack and calculate size
                scc_size = 0
                scc_members = set()
                while True:
                    member_id = stack.pop()
                    bookkeeping[member_id].on_stack = False
                    scc_size += self.get_object_summary(member_id).size
                    scc_members.add(member_id)
                    obj_id_to_scc[member_id] = entry.index
                    if member_id == obj_id:
                        break
                scc_sizes[entry.index] = scc_size

                subtree_size = scc_size + sum(
                    scc_sizes[child_scc] for child_scc in child_sccs
                )
                for member_id in scc_members:
                    data = t.get(_pack_id(member_id), db=self._primary_db)
                    assert data is not None
                    record = ObjectRecord[int].model_validate_json(data)
                    record.subtree_size = subtree_size
                    t.put(
                        _pack_id(member_id),
                        record.model_dump_json().encode(),
                        db=self._primary_db,
                    )

                return child_sccs.union({entry.index})
            return child_sccs

        cursor = t.cursor(db=self._primary_db)
        for key in cursor.iternext(keys=True, values=False):
            obj_id = _unpack_id(key)
            if obj_id not in bookkeeping:
                run_with_long_stack(strongconnect(obj_id))
