from midden.long_stack import run_with_long_stack
from collections.abc import Generator, Callable
from typing import TypeVar, Generic, Iterable
from dataclasses import dataclass

NodeAccT = TypeVar("NodeAccT")
SCCAccT = TypeVar("SCCAccT")
NodeIdT = TypeVar("NodeIdT")
NodeT = TypeVar("NodeT")


@dataclass(slots=True)
class _StackEntry(Generic[NodeIdT, NodeAccT]):
    id: NodeIdT
    acc: NodeAccT


@dataclass(slots=True)
class _BookkeepingEntry:
    index: int
    lowlink: int
    on_stack: bool


@dataclass(slots=True)
class _NodeInfo(Generic[NodeIdT, NodeAccT]):
    id: NodeIdT
    acc: NodeAccT


_EMPTY_SENTINEL = object()


class GraphSCCVisitor(Generic[NodeT, NodeIdT, NodeAccT, SCCAccT]):
    """An implementation or Tarjan's algorithm for finding strongly connected components (SCCs)
    in a directed graph, with support for accumulating values across nodes and SCCs.
    """

    def iterate_nodes(
        self, already_visited: Callable[[NodeIdT], bool]
    ) -> Iterable[NodeT]:
        """Override this to iterate over all nodes in the graph."""
        raise NotImplementedError

    def get_node_id(self, node: NodeT) -> NodeIdT:
        """Override this to get node information for a given node."""
        raise NotImplementedError

    def get_node_acc(self, node: NodeT) -> NodeAccT:
        """Override this to get the initial accumulator value for a given node."""
        raise NotImplementedError

    def get_successors(self, node: NodeT) -> Iterable[NodeT]:
        """Override this to get the successors of a given node."""
        raise NotImplementedError

    def accumulate_node_values(self, v1: NodeAccT, v2: NodeAccT) -> NodeAccT:
        """Override this to define how to accumulate two node values. This is used to combine values of nodes in the same SCC."""
        raise NotImplementedError

    def accumulate(
        self,
        node_acc: NodeAccT,
        this_scc: int,
        scc_values: Iterable[tuple[int, SCCAccT]],
    ) -> SCCAccT:
        """Override this to define how to accumulate node and child SCC values into an SCC value."""
        raise NotImplementedError

    def emit_result(self, node_id: NodeIdT, scc_acc: SCCAccT):
        """Override this to do something with the result for each node after processing. node_id is the ID of the node, and scc_acc is the accumulated value for its SCC."""
        raise NotImplementedError


def visit_sccs(visitor: GraphSCCVisitor[NodeT, NodeIdT, NodeAccT, SCCAccT]) -> None:
    """Run Tarjan's algorithm. For each SCC, visitor.emit_result will be called for each member of the SCC with the accumulated value for that SCC."""

    bookkeeping: dict[NodeIdT, _BookkeepingEntry] = {}
    stack: list[_StackEntry[NodeIdT, NodeAccT]] = []
    index = 0
    scc_accs: dict[int, SCCAccT] = {}

    def strongconnect(
        obj: NodeT,
    ) -> Generator[
        set[int], None, set[int]
    ]:  # Returns set of SCCs that are children of this node
        nonlocal index
        obj_id = visitor.get_node_id(obj)
        entry = _BookkeepingEntry(index=index, lowlink=index, on_stack=True)
        bookkeeping[obj_id] = entry
        stack.append(_StackEntry(id=obj_id, acc=visitor.get_node_acc(obj)))
        index += 1
        reachable_sccs: set[int] = set()

        for successor in visitor.get_successors(obj):
            # Don't include references from modules in the graph, since they create huge SCCs that aren't interesting
            successor_id = visitor.get_node_id(successor)
            bookkeeping_entry = bookkeeping.get(successor_id)
            if bookkeeping_entry is None:
                child_sccs = yield strongconnect(successor)
                reachable_sccs.update(child_sccs)
                entry.lowlink = min(entry.lowlink, bookkeeping[successor_id].lowlink)
            elif bookkeeping_entry.on_stack:
                entry.lowlink = min(entry.lowlink, bookkeeping_entry.index)
            else:
                # We use lowlink to hold SCC ids for nodes that have already been fully processed,
                # since we won't be visiting them again and we need some way to identify
                # which SCC they belong to when we accumulate results for parent SCCs
                reachable_sccs.add(bookkeeping[successor_id].lowlink)

        if entry.index == entry.lowlink:
            # Found an SCC root, pop the stack and calculate size
            scc = entry.index

            acc = _EMPTY_SENTINEL
            scc_members: set[NodeIdT] = set()
            while True:
                member = stack.pop()
                scc_members.add(member.id)
                acc = (
                    member.acc
                    if acc is _EMPTY_SENTINEL
                    else visitor.accumulate_node_values(acc, member.acc)
                )
                bookkeeping_item = bookkeeping[member.id]
                bookkeeping_item.on_stack = False
                bookkeeping_item.lowlink = scc  # Set lowlink to SCC index for all members of the SCC, so we can identify which SCC they belong to later

                if member.id == obj_id:
                    break
            acc = visitor.accumulate(
                acc,
                scc,
                ((child_scc, scc_accs[child_scc]) for child_scc in reachable_sccs),
            )
            scc_accs[scc] = acc

            for member_id in scc_members:
                visitor.emit_result(member_id, acc)
            reachable_sccs.add(scc)

        return reachable_sccs

    for obj in visitor.iterate_nodes(bookkeeping.__contains__):
        obj_id = visitor.get_node_id(obj)
        if obj_id not in bookkeeping:
            run_with_long_stack(strongconnect(obj))
