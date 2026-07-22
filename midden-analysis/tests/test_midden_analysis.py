import json
from pathlib import Path

import pytest
from midden_analysis import EstimatorPrecision, HeapDumpExplorer

TEST_DATA = [
    json.dumps(x).encode("utf-8")
    for x in [
        {"id": 1, "type": "builtins.module", "references": [2, 3, 5], "size": 10},
        {"id": 2, "type": "builtins.list", "references": [4], "size": 20},
        {"id": 3, "type": "builtins.dict", "references": [4], "size": 30},
        {"id": 4, "type": "builtins.str", "references": [], "size": 40},
        {"id": 5, "type": "builtins.module", "references": [], "size": 1000},
    ]
]


@pytest.mark.parametrize(
    ("estimator_precision", "margin_for_error"),
    [
        pytest.param(EstimatorPrecision.Exact, 0, id="exact"),
        pytest.param(EstimatorPrecision.High, 15, id="high"),
        pytest.param(EstimatorPrecision.Medium, 30, id="medium"),
        pytest.param(EstimatorPrecision.Low, 60, id="low"),
        pytest.param(EstimatorPrecision.NoEstimates, None, id="no_estimates"),
    ],
)
def test_heap_dump_explorer(
    tmp_path: Path,
    estimator_precision: EstimatorPrecision,
    margin_for_error: int | None,
):
    explorer = HeapDumpExplorer(str(tmp_path))
    explorer.import_lines(TEST_DATA, estimator_precision=estimator_precision)

    obj_1 = explorer.get_object(1)
    assert obj_1 is not None
    assert obj_1.type == "builtins.module"
    assert obj_1.size == 10
    if margin_for_error is not None:
        assert obj_1.subtree_size is not None
        assert 100 - margin_for_error <= obj_1.subtree_size <= 100 + margin_for_error
    assert {ref.id for ref in obj_1.references} == {2, 3, 5}

    obj_6 = explorer.get_object(6)
    assert obj_6 is None

    path = explorer.find_path_between_objects(1, 4, {2})
    assert path is not None
    assert [obj.id for obj in path] == [1, 3, 4]

    path_to_5 = explorer.find_path_between_objects(1, 5, set())
    assert path_to_5 is None

    assert explorer.get_count_for_type("builtins.module") == 2
    assert explorer.get_count_for_type("All Types") == 5
    type_summaries = dict(explorer.get_type_summaries())
    assert len(type_summaries) == 5
    assert type_summaries["builtins.module"].count == 2
    assert type_summaries["builtins.module"].total_size == 1010
    assert type_summaries["All Types"].count == 5
    assert type_summaries["All Types"].total_size == 1100

    assert len(explorer.get_objects_by_type("builtins.module", page=None)) == 2
    assert len(explorer.get_objects_by_type("builtins.list", page=None)) == 1
    object_by_size = explorer.get_objects_by_type_ordered_by_size(
        "builtins.module", subtree_size=False, page=None
    )
    assert len(object_by_size) == 2
    assert object_by_size[0].id == 5
    assert object_by_size[1].id == 1

    if estimator_precision != EstimatorPrecision.NoEstimates:
        assert explorer.find_largest_common_reachable_object(2, 3) == 4
        assert explorer.find_largest_common_reachable_object(1, 3) == 3
        assert explorer.find_largest_common_reachable_object(2, 5) is None

    if estimator_precision in (EstimatorPrecision.Exact, EstimatorPrecision.High):
        # id 1 subtree=100; largest strict descendant is id 3 (subtree=70)
        assert explorer.find_largest_strict_descendant(1) == 3
        # id 4 has no descendants
        assert explorer.find_largest_strict_descendant(4) is None

        # largest ancestor of id 4 (subtree=40) is id 1 (subtree=100)
        assert explorer.find_largest_ancestor(4) == 1
        # id 1 has no referrers, so it is its own largest ancestor
        assert explorer.find_largest_ancestor(1) == 1
