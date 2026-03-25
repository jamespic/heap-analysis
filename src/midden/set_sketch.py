"""A data structure that compactly represents a sketch of a set, to allow quickly asking "is this sketch a subset of that sketch?",
and quickly get an answer of "no", or "maybe"."""

from array import array
import xxhash


class SetSketch:
    """A MinHash-based sketch for probabilistic subset testing.

    Given two sets A and B, if A ⊆ B then for every hash function h,
    min(h(x) for x in B) <= min(h(x) for x in A), because B contains
    all elements of A (and possibly more).

    If A is NOT a subset of B, there's a small probability we still say
    "maybe" (false positive), controlled by the number of hash functions k.
    The false positive rate is approximately (J(A,B))^k where J is the
    Jaccard similarity, but in practice for non-subsets it decreases
    exponentially with k.

    Args:
        num_hashes: Number of independent hash functions (k). Higher values
            reduce false positive rate but increase sketch size. Each hash
            uses 8 bytes, so sketch size = 8 * num_hashes bytes.
        from_bytes: Optional bytes to initialize the sketch from. num_hashes is ignored
    """

    def __init__(self, num_hashes: int = 8, from_bytes: bytes | None = None):
        self.max_hash = (1 << 64) - 1
        # Initialize all registers to max value (empty set)
        if from_bytes is not None:
            self._registers = array("Q")
            self._registers.frombytes(from_bytes)
            self.num_hashes = len(self._registers)
        else:
            self.num_hashes = num_hashes
            self._registers: array = array("Q", [self.max_hash] * num_hashes)

    def _hash(self, item: bytes, i: int) -> int:
        """Compute the i-th hash function on the given item.

        Uses xxHash with different seeds to simulate independent hash functions.
        """
        h = xxhash.xxh64(seed=i)
        h.update(item)
        # Take first 8 bytes as a 64-bit unsigned integer
        return h.intdigest()

    def _convert_item_to_bytes(self, item) -> bytes:
        """Convert an item to bytes for hashing."""
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        else:
            return repr(item).encode("utf-8")

    def add(self, item) -> "SetSketch":
        """Add an item to the sketch.

        For each hash function, update the register to hold the minimum
        hash value seen so far.
        """
        item_bytes = self._convert_item_to_bytes(item)
        for i in range(self.num_hashes):
            h = self._hash(item_bytes, i)
            if h < self._registers[i]:
                self._registers[i] = h
        return self

    def add_all(self, items) -> "SetSketch":
        """Add multiple items to the sketch."""
        for item in items:
            self.add(item)
        return self

    def is_subset_of(self, other: "SetSketch") -> bool:
        """Test whether this sketch's set is possibly a subset of other's set.

        Returns:
            False: Definitely NOT a subset (no false negatives).
            True:  MAYBE a subset (may be a false positive).

        The invariant: if A ⊆ B, then for all i, min_hash_i(B) <= min_hash_i(A).
        Equivalently, other._registers[i] <= self._registers[i] for all i.

        If any register in `other` is strictly greater than the corresponding
        register in `self`, then other is missing an element that contributed
        the minimum hash in self — so self is definitely not a subset of other.
        """
        if self.num_hashes != other.num_hashes:
            raise ValueError("Cannot compare sketches with different parameters")
        for i in range(self.num_hashes):
            if other._registers[i] > self._registers[i]:
                return False
        return True

    def to_bytes(self) -> bytes:
        """Serialize the sketch to bytes."""
        return self._registers.tobytes()

    def estimated_false_positive_rate(self, other: "SetSketch") -> float:
        """Estimate the false positive rate by computing what fraction of
        registers are consistent with the subset relationship.

        This gives a rough sense of confidence in a "maybe" answer.
        """
        if self.num_hashes != other.num_hashes:
            raise ValueError("Cannot compare sketches with different parameters")
        consistent = sum(
            1
            for i in range(self.num_hashes)
            if other._registers[i] <= self._registers[i]
        )
        # If all are consistent, estimate FP rate as (consistent/total)^k
        # but really this is just the fraction consistent
        return consistent / self.num_hashes

    @property
    def is_empty(self) -> bool:
        """Check if no items have been added."""
        return all(r == self.max_hash for r in self._registers)

    def __repr__(self) -> str:
        return f"SetSketch(num_hashes={self.num_hashes})"
