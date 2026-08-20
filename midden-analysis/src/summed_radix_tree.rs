use std::{array, rc::Rc};

use xxhash_rust::xxh3::Xxh3Default;

const FANOUT: usize = 8;

/// Number of values represented by each leaf and number of children per branch.
///
/// A small fanout keeps the tree shallow while making each union inexpensive:
/// branches can merge their children independently and share unchanged `Rc`s.

/// Persistent radix tree used for exact subtree-size unions and size calculation.
///
/// The tree stores a sparse mapping from positions to values. A position's
/// value is the maximum value contributed by any input tree, so `union` can be
/// used to combine the exact sets of objects reachable from several roots.
/// Trees are persistent: updates create only the nodes on the changed paths
/// and share the rest with their inputs. Each node caches a hash and the
/// total of all values in its subtree, so equality checks and total queries are
/// cheap. The tree is sparse: empty children are represented by a shared empty
/// tree, and leaves are only created for positions that have non-zero values.
///
/// Leaves hold up to `FANOUT` adjacent positions. A branch at `level = n`
/// groups `FANOUT.pow(n)` positions into each child, with child zero covering
/// the lowest range. Empty children are represented by the shared `Empty` tree.
#[derive(Debug)]
pub enum SummedRadixTree {
    Empty,
    Leaf {
        values: [u64; FANOUT],
        hash: u128,
        total: u64,
    },
    Branch {
        // `level` is the number of radix digits below this branch. A level-1
        // branch points to leaves; higher levels point to smaller branches.
        level: u8,
        children: [Rc<SummedRadixTree>; FANOUT],
        // Cached values make equality checks and aggregate queries cheap.
        hash: u128,
        total: u64,
    },
}

thread_local! {
    static EMPTY: Rc<SummedRadixTree> = Rc::new(SummedRadixTree::Empty);
}

impl SummedRadixTree {
    /// Return the shared empty tree.
    pub fn new() -> Rc<Self> {
        EMPTY.with(|empty| empty.clone())
    }

    /// Read the value stored at one position.
    pub fn get_value(&self, position: usize) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Leaf { values, .. } => {
                if position >= FANOUT {
                    0
                } else {
                    values[position]
                }
            }
            Self::Branch {
                level, children, ..
            } => {
                let items_per_element = FANOUT.pow(*level as u32);
                let child_index = position / items_per_element;
                if child_index >= FANOUT {
                    0
                } else {
                    let child_offset = position % items_per_element;
                    children[child_index].get_value(child_offset)
                }
            }
        }
    }

    #[allow(unused)] // Used in test configurations, but not in production code, so allow it to be unused.
    /// Return whether a position has a non-zero value.
    pub fn contains(&self, position: usize) -> bool {
        self.get_value(position) > 0
    }

    /// Return a structural hash used for cheap equality checks during unions.
    ///
    /// Hashes are cached because unions commonly compare many subtrees. A hash
    /// match is only a fast path, not a proof of equality; callers relying on
    /// collision resistance should treat the result as an optimisation.
    pub fn unique_hash(&self) -> u128 {
        match self {
            Self::Empty => 0,
            Self::Leaf { hash, .. } => *hash,
            Self::Branch { hash, .. } => *hash,
        }
    }

    /// Return the sum of all values stored in the tree.
    ///
    /// This is cached in every non-empty node and is therefore independent of
    /// the cost of walking the tree.
    pub fn total(&self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Leaf { total, .. } => *total,
            Self::Branch { total, .. } => *total,
        }
    }

    /// Return a tree with one position updated to the maximum-seen value.
    pub fn add(self: &Rc<Self>, position: usize, value: u64) -> Rc<Self> {
        if self.get_value(position) == value {
            self.clone()
        } else {
            let single_position_set = Self::_with_single_position_set(position, value);
            self.union(&Rc::new(single_position_set))
        }
    }

    /// Union two trees by taking the elementwise maximum.
    ///
    /// When the trees have different heights, the shorter tree belongs in the
    /// lowest child of the taller one because both represent ranges starting
    /// at position zero. This is also why positions are encoded from the
    /// least significant radix digits outward in `_with_single_position_set`.
    pub fn union(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Self> {
        match (self.as_ref(), other.as_ref()) {
            (Self::Empty, _) => other.clone(),
            (_, Self::Empty) => self.clone(),
            (s, o) if s.unique_hash() == o.unique_hash() => {
                if Rc::strong_count(self) >= Rc::strong_count(other) {
                    self.clone()
                } else {
                    other.clone()
                }
            }
            (Self::Leaf { .. }, Self::Branch { .. }) => {
                other.union(self) // Let the bigger tree handle merging
            }
            (
                Self::Branch {
                    level, children, ..
                },
                Self::Leaf { .. },
            ) => {
                let mut new_children = children.clone();
                new_children[0] = new_children[0].clone().union(other);
                let hash = Self::_calculate_branch_hash(&new_children);
                let total = new_children.iter().map(|child| child.total()).sum();
                Rc::new(Self::Branch {
                    level: *level,
                    children: new_children,
                    hash,
                    total,
                })
            }
            (
                Self::Branch {
                    level: l1,
                    children: c1,
                    ..
                },
                Self::Branch {
                    level: l2,
                    children: c2,
                    ..
                },
            ) => {
                match l1.cmp(l2) {
                    std::cmp::Ordering::Greater => {
                        let mut new_children = c1.clone();
                        new_children[0] = new_children[0].clone().union(other);
                        let hash = Self::_calculate_branch_hash(&new_children);
                        let total = new_children.iter().map(|child| child.total()).sum();
                        Rc::new(Self::Branch {
                            level: *l1,
                            children: new_children,
                            hash,
                            total,
                        })
                    }
                    std::cmp::Ordering::Less => {
                        other.union(self) // Let the bigger tree handle merging
                    }
                    std::cmp::Ordering::Equal => {
                        let new_children = array::from_fn(|i| c1[i].clone().union(&c2[i]));
                        let hash = Self::_calculate_branch_hash(&new_children);
                        let total = new_children.iter().map(|child| child.total()).sum();
                        Rc::new(Self::Branch {
                            level: *l1,
                            children: new_children,
                            hash,
                            total,
                        })
                    }
                }
            }
            (Self::Leaf { values: v1, .. }, Self::Leaf { values: v2, .. }) => {
                if v1 == v2 {
                    self.clone()
                } else {
                    let new_values = array::from_fn(|i| v1[i].max(v2[i]));
                    let hash = Self::_calculate_leaf_hash(&new_values);
                    let total = new_values.iter().sum();
                    Rc::new(Self::Leaf {
                        values: new_values,
                        hash,
                        total,
                    })
                }
            }
        }
    }

    fn _with_single_position_set(position: usize, value: u64) -> Self {
        let leaf_index = position / FANOUT;
        let leaf_position = position % FANOUT;

        let mut leaf_values = [0u64; FANOUT];
        leaf_values[leaf_position] = value;
        let mut result = Self::Leaf {
            values: leaf_values,
            hash: Self::_calculate_leaf_hash(&leaf_values),
            total: value,
        };

        // Make parent branch nodes for the radix digits above this leaf.
        let mut current_level = 1;
        let mut child_index = leaf_index;
        while child_index > 0 {
            let parent_index = child_index % FANOUT;
            child_index /= FANOUT;
            let mut children: [Rc<Self>; FANOUT] = array::from_fn(|_| SummedRadixTree::new());
            children[parent_index] = Rc::new(result);
            let hash = Self::_calculate_branch_hash(&children);
            let new_branch = Self::Branch {
                level: current_level,
                children,
                hash,
                total: value,
            };
            result = new_branch;
            current_level += 1;
        }

        result
    }

    fn _calculate_branch_hash(children: &[Rc<Self>; FANOUT]) -> u128 {
        let mut hasher = Xxh3Default::new();
        for child in children {
            let child_hash = child.unique_hash();
            hasher.update(&child_hash.to_le_bytes());
        }
        hasher.digest128()
    }

    fn _calculate_leaf_hash(values: &[u64; FANOUT]) -> u128 {
        let mut hasher = Xxh3Default::new();
        for bit in values {
            hasher.update(&bit.to_le_bytes());
        }
        hasher.digest128()
    }

    fn _estimate_size(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Leaf { .. } => size_of::<Self>(),
            Self::Branch { children, .. } => {
                size_of::<Self>()
                    + children
                        .iter()
                        .map(|child| child._estimate_size_fudging_refcounts())
                        .sum::<usize>()
            }
        }
    }

    pub fn _estimate_size_fudging_refcounts(self: &Rc<Self>) -> usize {
        (self._estimate_size() / Rc::strong_count(self)) + size_of::<Rc<()>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let tree = SummedRadixTree::new();
        assert!(!tree.contains(5));
        let tree = tree.add(5, 10);
        assert!(tree.contains(5));
        assert_eq!(tree.total(), 10);

        let tree2 = SummedRadixTree::new().add(5, 15).add(10, 20);
        let union_tree = tree.union(&tree2);
        assert!(union_tree.contains(5));
        assert!(union_tree.contains(10));
        assert_eq!(union_tree.total(), 35);
    }

    fn compare_with_naive_implementation(values: Vec<Vec<(usize, u64)>>) {
        let mut naive_map = std::collections::HashMap::new();
        let mut tree = SummedRadixTree::new();

        for subset in values {
            let mut inner_tree = SummedRadixTree::new();
            for (element, value) in subset {
                inner_tree = inner_tree.add(element, value);
                naive_map.insert(element, value);
            }
            tree = tree.union(&inner_tree);
        }

        for (element, value) in &naive_map {
            assert!(tree.contains(*element));
            assert_eq!(tree.get_value(*element), *value);
        }
        assert!(tree.total() == naive_map.values().sum::<u64>());
    }

    macro_rules! generate_tests {
        ($($name:ident: $values:expr),+) => {
            $(
                #[test]
                fn $name() {
                    compare_with_naive_implementation($values);
                }
            )*
        };
    }

    generate_tests!(
        test_single_element: vec![vec![(5, 10)]],
        test_multiple_elements: vec![vec![(5, 10), (10, 20), (15, 30)]],
        test_overlapping_elements: vec![vec![(5, 10)], vec![(5, 15)], vec![(10, 20)]],
        test_large_numbers: vec![vec![(1000, 1), (2000, 2)], vec![(1000, 3), (3000, 4)]],
        test_empty_tree: vec![],
        test_powers_of_two: vec![vec![(0, 1), (8, 2), (64, 3), (512, 4)]],
        test_descending_powers_of_two: vec![vec![(512, 4), (64, 3), (8, 2), (0, 1)]],
        test_one_to_ten: vec![(0..10).map(|i| (i, i as u64 + 1)).collect()],
        test_descending_one_to_ten: vec![(0..10).rev().map(|i| (i, i as u64 + 1)).collect()]
    );
}
