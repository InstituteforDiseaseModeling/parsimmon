from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sciris as sc

from parsimmon.cache import SimFileCache
from parsimmon.results import SimResult, _SimEntry


def _make_result(groups_spec):
    values = []
    metadata_list = []
    sim_id = 1
    for group_name, count, pars in groups_spec:
        for group_id in range(1, count + 1):
            values.append({"value": sim_id})
            metadata_list.append({
                "parameter_set": "test_set",
                "group": group_name,
                "sim_id": sim_id,
                "group_id": group_id,
                "pars": sc.objdict(pars),
            })
            sim_id += 1
    return SimResult.from_values(values, metadata_list)


def _make_lazy_entry(metadata, return_value=None):
    mock_backend = MagicMock()
    mock_backend.load.return_value = return_value if return_value is not None else {"result": 42}
    entry = _SimEntry(
        metadata=metadata,
        cache_key="abc123",
        backend=mock_backend,
    )
    return entry, mock_backend


@sc.timer()
def test_iter_and_len():
    sr = _make_result([("Baseline", 3, {"beta": 0.5})])

    assert len(sr) == 3, f"Expected len 3, got {len(sr)}"

    values = list(sr)
    assert len(values) == 3, f"Expected 3 items from iteration, got {len(values)}"

    for i, v in enumerate(values, start=1):
        assert isinstance(v, dict), f"Expected dict value, got {type(v)}"
        assert v["value"] == i, f"Expected value {i}, got {v['value']}"


@sc.timer()
def test_getitem_int():
    sr = _make_result([("A", 3, {"x": 1})])

    assert sr[0] == {"value": 1}, f"sr[0] should be first value, got {sr[0]}"
    assert sr[1] == {"value": 2}, f"sr[1] should be second value, got {sr[1]}"
    assert sr[2] == {"value": 3}, f"sr[2] should be third value, got {sr[2]}"
    assert sr[-1] == {"value": 3}, f"sr[-1] should be last value, got {sr[-1]}"


@sc.timer()
def test_getitem_slice():
    sr = _make_result([("A", 5, {"x": 1})])

    sliced = sr[1:4]
    assert isinstance(sliced, SimResult), f"Slice should return SimResult, got {type(sliced)}"
    assert len(sliced) == 3, f"Slice [1:4] of 5 should have len 3, got {len(sliced)}"

    sliced2 = sr[:2]
    assert len(sliced2) == 2, f"Slice [:2] should have len 2, got {len(sliced2)}"

    values = list(sliced)
    assert values[0] == {"value": 2}, f"First sliced value should be 2, got {values[0]}"


@sc.timer()
def test_bool():
    empty = SimResult([])
    assert not empty, "Empty SimResult should be falsy"

    non_empty = _make_result([("A", 1, {})])
    assert non_empty, "Non-empty SimResult should be truthy"


@sc.timer()
def test_attr_group_access():
    sr = _make_result([
        ("Baseline", 2, {"beta": 0.5}),
        ("Treatment", 3, {"beta": 0.8}),
    ])

    baseline = sr.Baseline
    assert isinstance(baseline, SimResult), f"Expected SimResult, got {type(baseline)}"
    assert len(baseline) == 2, f"Expected 2 Baseline entries, got {len(baseline)}"

    treatment = sr.Treatment
    assert len(treatment) == 3, f"Expected 3 Treatment entries, got {len(treatment)}"

    for meta in baseline.metadata:
        assert meta["group"] == "Baseline", f"Expected group 'Baseline', got {meta['group']}"


@sc.timer()
def test_attr_parameter_set_access():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {"parameter_set": "set_A", "group": "Control", "sim_id": 1, "group_id": 1, "pars": sc.objdict({})},
        {"parameter_set": "set_A", "group": "Control", "sim_id": 2, "group_id": 2, "pars": sc.objdict({})},
        {"parameter_set": "set_B", "group": "Control", "sim_id": 3, "group_id": 1, "pars": sc.objdict({})},
        {"parameter_set": "set_B", "group": "Control", "sim_id": 4, "group_id": 2, "pars": sc.objdict({})},
    ]
    sr = SimResult.from_values(values, metadata_list)

    set_a = sr.set_A
    assert isinstance(set_a, SimResult), f"Expected SimResult, got {type(set_a)}"
    assert len(set_a) == 2, f"Expected 2 entries in set_A, got {len(set_a)}"

    set_b = sr.set_B
    assert len(set_b) == 2, f"Expected 2 entries in set_B, got {len(set_b)}"

    for meta in set_a.metadata:
        assert meta["parameter_set"] == "set_A", f"Expected set_A, got {meta['parameter_set']}"


@sc.timer()
def test_attr_chained():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {"parameter_set": "tipping_point", "group": "Baseline", "sim_id": 1, "group_id": 1, "pars": sc.objdict({})},
        {"parameter_set": "tipping_point", "group": "Baseline", "sim_id": 2, "group_id": 2, "pars": sc.objdict({})},
        {"parameter_set": "control", "group": "Baseline", "sim_id": 3, "group_id": 1, "pars": sc.objdict({})},
        {"parameter_set": "control", "group": "Baseline", "sim_id": 4, "group_id": 2, "pars": sc.objdict({})},
    ]
    sr = SimResult.from_values(values, metadata_list)

    chained = sr.tipping_point.Baseline
    assert isinstance(chained, SimResult), f"Expected SimResult, got {type(chained)}"
    assert len(chained) == 2, f"Expected 2 entries after chaining, got {len(chained)}"

    for meta in chained.metadata:
        assert meta["parameter_set"] == "tipping_point"
        assert meta["group"] == "Baseline"


@sc.timer()
def test_attr_missing_raises():
    sr = _make_result([("Baseline", 2, {})])

    with pytest.raises(AttributeError, match="no attribute 'NonExistent'"):
        _ = sr.NonExistent

    with pytest.raises(AttributeError, match="Baseline"):
        _ = sr.NonExistent


@sc.timer()
def test_groups_property():
    sr = _make_result([
        ("Baseline", 2, {"beta": 0.5}),
        ("Treatment", 3, {"beta": 0.8}),
    ])

    groups = sr.groups
    assert "Baseline" in groups, f"Expected 'Baseline' in groups, got {list(groups.keys())}"
    assert "Treatment" in groups, f"Expected 'Treatment' in groups, got {list(groups.keys())}"

    assert isinstance(groups["Baseline"], SimResult)
    assert isinstance(groups["Treatment"], SimResult)
    assert len(groups["Baseline"]) == 2, f"Expected 2, got {len(groups['Baseline'])}"
    assert len(groups["Treatment"]) == 3, f"Expected 3, got {len(groups['Treatment'])}"

    total = sum(len(v) for v in groups.values())
    assert total == len(sr), f"Groups total {total} != original len {len(sr)}"


@sc.timer()
def test_group_by_key():
    sr = _make_result([
        ("A", 2, {"beta": 0.5}),
        ("B", 3, {"beta": 0.8}),
    ])

    grouped = sr.group("beta")
    assert 0.5 in grouped, f"Expected key 0.5 in grouped, got {list(grouped.keys())}"
    assert 0.8 in grouped, f"Expected key 0.8 in grouped, got {list(grouped.keys())}"
    assert len(grouped[0.5]) == 2, f"Expected 2 entries for beta=0.5, got {len(grouped[0.5])}"
    assert len(grouped[0.8]) == 3, f"Expected 3 entries for beta=0.8, got {len(grouped[0.8])}"


@sc.timer()
def test_group_nested_key():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {"parameter_set": "test_set", "group": "A", "sim_id": 1, "group_id": 1, "pars": {"model": {"rate": 0.1}}},
        {"parameter_set": "test_set", "group": "A", "sim_id": 2, "group_id": 2, "pars": {"model": {"rate": 0.1}}},
        {"parameter_set": "test_set", "group": "B", "sim_id": 3, "group_id": 1, "pars": {"model": {"rate": 0.9}}},
        {"parameter_set": "test_set", "group": "B", "sim_id": 4, "group_id": 2, "pars": {"model": {"rate": 0.9}}},
    ]
    sr = SimResult.from_values(values, metadata_list)

    grouped = sr.group("model.rate")
    assert 0.1 in grouped, f"Expected 0.1 in grouped, got {list(grouped.keys())}"
    assert 0.9 in grouped, f"Expected 0.9 in grouped, got {list(grouped.keys())}"
    assert len(grouped[0.1]) == 2
    assert len(grouped[0.9]) == 2


@sc.timer()
def test_filter_by_key_value():
    sr = _make_result([
        ("A", 2, {"beta": 0.5}),
        ("B", 3, {"beta": 0.8}),
    ])

    filtered = sr.filter("beta", 0.5)
    assert isinstance(filtered, SimResult)
    assert len(filtered) == 2, f"Expected 2 filtered entries, got {len(filtered)}"

    for meta in filtered.metadata:
        assert meta["pars"]["beta"] == 0.5, f"Unexpected beta: {meta['pars']['beta']}"


@sc.timer()
def test_filter_by_dotted_key():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {"parameter_set": "test_set", "group": "A", "sim_id": 1, "group_id": 1, "pars": {"model": {"rate": 0.1}}},
        {"parameter_set": "test_set", "group": "A", "sim_id": 2, "group_id": 2, "pars": {"model": {"rate": 0.5}}},
        {"parameter_set": "test_set", "group": "B", "sim_id": 3, "group_id": 1, "pars": {"model": {"rate": 0.1}}},
        {"parameter_set": "test_set", "group": "B", "sim_id": 4, "group_id": 2, "pars": {"model": {"rate": 0.5}}},
    ]
    sr = SimResult.from_values(values, metadata_list)

    filtered = sr.filter("model.rate", 0.1)
    assert len(filtered) == 2, f"Expected 2, got {len(filtered)}"

    for meta in filtered.metadata:
        assert meta["pars"]["model"]["rate"] == 0.1


@sc.timer()
def test_filter_by_metadata_key():
    sr = _make_result([
        ("Baseline", 2, {"x": 1}),
        ("Treatment", 3, {"x": 2}),
    ])

    filtered = sr.filter("group", "Baseline")
    assert len(filtered) == 2, f"Expected 2, got {len(filtered)}"

    for meta in filtered.metadata:
        assert meta["group"] == "Baseline"


@sc.timer()
def test_filter_by_predicate():
    sr = _make_result([
        ("Low", 2, {"beta": 0.1}),
        ("High", 3, {"beta": 0.8}),
    ])

    filtered = sr.filter(lambda pars, meta: pars["beta"] > 0.3)
    assert len(filtered) == 3, f"Expected 3, got {len(filtered)}"

    for meta in filtered.metadata:
        assert meta["pars"]["beta"] > 0.3

    filtered2 = sr.filter(lambda pars, meta: meta["group"] == "Low" and pars["beta"] < 0.5)
    assert len(filtered2) == 2


@sc.timer()
def test_pars_property():
    sr = _make_result([
        ("A", 2, {"alpha": 0.1, "beta": 0.5}),
        ("B", 1, {"alpha": 0.2, "beta": 0.8}),
    ])

    pars = sr.pars
    assert isinstance(pars, list), f"Expected list, got {type(pars)}"
    assert len(pars) == 3, f"Expected 3 pars dicts, got {len(pars)}"

    assert pars[0]["beta"] == 0.5
    assert pars[1]["beta"] == 0.5
    assert pars[2]["beta"] == 0.8


@sc.timer()
def test_metadata_property():
    sr = _make_result([
        ("Baseline", 2, {"beta": 0.5}),
    ])

    metadata = sr.metadata
    assert isinstance(metadata, list)
    assert len(metadata) == 2

    for i, meta in enumerate(metadata, start=1):
        assert "group" in meta
        assert "sim_id" in meta
        assert "pars" in meta
        assert meta["group"] == "Baseline"
        assert meta["sim_id"] == i


@sc.timer()
def test_lazy_no_load_on_filter():
    entry1, mock1 = _make_lazy_entry({"group": "A", "pars": {"x": 1}})
    entry2, mock2 = _make_lazy_entry({"group": "B", "pars": {"x": 2}})
    sr = SimResult([entry1, entry2])

    _ = sr.filter("group", "A")

    mock1.load.assert_not_called()
    mock2.load.assert_not_called()


@sc.timer()
def test_lazy_no_load_on_groups():
    entry1, mock1 = _make_lazy_entry({"group": "A", "pars": {"x": 1}})
    entry2, mock2 = _make_lazy_entry({"group": "B", "pars": {"x": 2}})
    sr = SimResult([entry1, entry2])

    _ = sr.groups

    mock1.load.assert_not_called()
    mock2.load.assert_not_called()


@sc.timer()
def test_lazy_no_load_on_pars():
    entry1, mock1 = _make_lazy_entry({"group": "A", "pars": {"x": 1}})
    entry2, mock2 = _make_lazy_entry({"group": "B", "pars": {"x": 2}})
    sr = SimResult([entry1, entry2])

    _ = sr.pars

    mock1.load.assert_not_called()
    mock2.load.assert_not_called()


@sc.timer()
def test_lazy_loads_on_iter():
    entry1, mock1 = _make_lazy_entry({"group": "A", "pars": {"x": 1}}, return_value={"result": 10})
    entry2, mock2 = _make_lazy_entry({"group": "B", "pars": {"x": 2}}, return_value={"result": 20})
    sr = SimResult([entry1, entry2])

    values = list(sr)

    mock1.load.assert_called_once_with("abc123")
    mock2.load.assert_called_once_with("abc123")
    assert values == [{"result": 10}, {"result": 20}], f"Unexpected values: {values}"


@sc.timer()
def test_lazy_loads_on_getitem_int():
    entry1, mock1 = _make_lazy_entry({"group": "A", "pars": {"x": 1}}, return_value={"result": 99})
    entry2, mock2 = _make_lazy_entry({"group": "B", "pars": {"x": 2}}, return_value={"result": 77})
    sr = SimResult([entry1, entry2])

    val = sr[0]

    mock1.load.assert_called_once_with("abc123")
    mock2.load.assert_not_called()
    assert val == {"result": 99}, f"Unexpected value: {val}"


@sc.timer()
def test_from_cache_file_path(tmp_path):
    cache = SimFileCache(tmp_path)
    import pickle

    def pkl_save(path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def pkl_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    cache._save = pkl_save
    cache._load = pkl_load

    cache.save("key001", {"result": 1}, {"group": "Baseline", "pars": {"beta": 0.5}})
    cache.save("key002", {"result": 2}, {"group": "Baseline", "pars": {"beta": 0.5}})
    cache.save("key003", {"result": 3}, {"group": "Treatment", "pars": {"beta": 0.8}})

    sr = SimResult.from_cache(tmp_path)
    assert len(sr) == 3, f"Expected 3 entries, got {len(sr)}"

    val = sr[0]
    assert val == {"result": 1}, f"Unexpected value: {val}"


@sc.timer()
def test_from_cache_backend_instance(tmp_path):
    import pickle

    def pkl_save(path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def pkl_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    cache = SimFileCache(tmp_path, save=pkl_save, load=pkl_load)
    cache.save("keyA", {"sim": "A"}, {"group": "Alpha", "pars": {"rate": 0.1}})
    cache.save("keyB", {"sim": "B"}, {"group": "Beta", "pars": {"rate": 0.9}})

    sr = SimResult.from_cache(cache)
    assert len(sr) == 2, f"Expected 2 entries, got {len(sr)}"

    vals = list(sr)
    assert len(vals) == 2, f"Expected 2 values from iteration, got {len(vals)}"


@sc.timer()
def test_repr():
    sr = _make_result([
        ("Baseline", 2, {}),
        ("Treatment", 3, {}),
    ])

    r = repr(sr)
    assert "SimResult" in r, f"repr should mention SimResult: {r!r}"
    assert "n=5" in r, f"repr should show n=5: {r!r}"
    assert "Baseline" in r, f"repr should mention 'Baseline': {r!r}"
    assert "Treatment" in r, f"repr should mention 'Treatment': {r!r}"

    empty_repr = repr(SimResult([]))
    assert "n=0" in empty_repr, f"Empty repr should show n=0: {empty_repr!r}"


@sc.timer()
def test_from_values_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        SimResult.from_values([1, 2, 3], [{"a": 1}, {"b": 2}])

    with pytest.raises(ValueError, match="same length"):
        SimResult.from_values([], [{"a": 1}])

    sr = SimResult.from_values([1, 2], [{"x": 1}, {"x": 2}])
    assert len(sr) == 2


if __name__ == "__main__":
    T = sc.timer()

    test_iter_and_len()
    test_getitem_int()
    test_getitem_slice()
    test_bool()
    test_attr_group_access()
    test_attr_parameter_set_access()
    test_attr_chained()
    test_attr_missing_raises()
    test_groups_property()
    test_group_by_key()
    test_group_nested_key()
    test_filter_by_key_value()
    test_filter_by_dotted_key()
    test_filter_by_metadata_key()
    test_filter_by_predicate()
    test_pars_property()
    test_metadata_property()
    test_lazy_no_load_on_filter()
    test_lazy_no_load_on_groups()
    test_lazy_no_load_on_pars()
    test_lazy_loads_on_iter()
    test_lazy_loads_on_getitem_int()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_from_cache_file_path(Path(tmp) / "cache1")
        test_from_cache_backend_instance(Path(tmp) / "cache2")

    test_repr()
    test_from_values_length_mismatch()

    T.toc()
