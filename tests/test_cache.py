import pickle
from pathlib import Path

import numpy as np
import pytest
import sciris as sc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from parsimmon.cache import (
    SimCacheBase,
    SimFileCache,
    compute_cache_key,
    find_project_root,
    hash_function_chain,
    hash_params,
    is_project_local,
)


@sc.timer()
def test_hash_deterministic():
    pars = {"a": 1, "b": "hello", "c": 3.14}
    h1 = hash_params(pars)
    h2 = hash_params(pars)
    assert h1 == h2, f"Expected identical hashes, got {h1!r} and {h2!r}"


@sc.timer()
def test_hash_key_order_invariant():
    h1 = hash_params({"a": 1, "b": 2})
    h2 = hash_params({"b": 2, "a": 1})
    assert h1 == h2, f"Key order should not matter; got {h1!r} != {h2!r}"


@sc.timer()
def test_hash_differentiates_values():
    h1 = hash_params({"a": 1})
    h2 = hash_params({"a": 2})
    assert h1 != h2, "Different values should produce different hashes"


@sc.timer()
def test_hash_differentiates_types():
    h_int = hash_params({"a": 1})
    h_str = hash_params({"a": "1"})
    assert h_int != h_str, "int 1 and str '1' should hash differently"


@sc.timer()
def test_hash_nested_dicts():
    h1 = hash_params({"outer": {"x": 1, "y": 2}})
    h2 = hash_params({"outer": {"y": 2, "x": 1}})
    assert h1 == h2, "Nested dict key order should not matter"

    h3 = hash_params({"outer": {"x": 1, "y": 99}})
    assert h1 != h3, "Different nested values should produce different hashes"


@sc.timer()
def test_hash_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    h1 = hash_params({"a": arr})
    h2 = hash_params({"a": arr.copy()})
    assert h1 == h2, "Arrays with same content should hash identically"

    arr_int = arr.astype(np.int32)
    h_int = hash_params({"a": arr_int})
    assert h1 != h_int, "Different dtype should produce different hash"

    arr_2d = arr.reshape(3, 1)
    h_2d = hash_params({"a": arr_2d})
    assert h1 != h_2d, "Different shape should produce different hash"


@sc.timer()
def test_hash_numpy_scalars():
    h_np = hash_params({"a": np.float64(1.0)})
    h_py = hash_params({"a": 1.0})
    assert h_np == h_py, "np.float64 and float should hash identically"

    h_np_int = hash_params({"a": np.int64(5)})
    h_py_int = hash_params({"a": 5})
    assert h_np_int == h_py_int, "np.int64 and int should hash identically"


@sc.timer()
def test_hash_lists_tuples():
    h_list = hash_params({"a": [1, 2, 3]})
    h_tuple = hash_params({"a": (1, 2, 3)})
    assert h_list != h_tuple, "list and tuple should hash differently even with same elements"

    h_list2 = hash_params({"a": [1, 2, 3]})
    assert h_list == h_list2, "Same list should hash identically"


@sc.timer()
def test_hash_none_and_bool():
    h_none = hash_params({"a": None})
    h_true = hash_params({"a": True})
    h_false = hash_params({"a": False})
    h_one = hash_params({"a": 1})
    h_zero = hash_params({"a": 0})

    assert len({h_none, h_true, h_false, h_one, h_zero}) == 5, (
        "None, True, False, int(1), int(0) should all have distinct hashes"
    )


@sc.timer()
def test_cache_key_truncated():
    key = compute_cache_key({"x": 1})
    assert len(key) == 16, f"Expected 16 chars, got {len(key)}: {key!r}"
    assert all(c in "0123456789abcdef" for c in key), f"Non-hex chars in key: {key!r}"


@sc.timer()
def test_cache_key_deterministic():
    pars = {"sim": {"dur": 365, "n_agents": 1000}, "beta": 0.5}
    k1 = compute_cache_key(pars)
    k2 = compute_cache_key(pars)
    assert k1 == k2, f"Cache key should be deterministic; got {k1!r} != {k2!r}"


@sc.timer()
def test_project_local_inside():
    project_root = _PROJECT_ROOT
    test_file = _PROJECT_ROOT / "src" / "parsimmon" / "cache.py"
    assert is_project_local(test_file, project_root), f"{test_file} should be project-local under {project_root}"


@sc.timer()
def test_project_local_site_packages():
    project_root = _PROJECT_ROOT
    sp_path = Path("/usr/lib/python3/site-packages/numpy/__init__.py")
    assert not is_project_local(sp_path, project_root), "site-packages path should not be project-local"


@sc.timer()
def test_project_local_venv(tmp_path):
    fake_root = tmp_path / "myproject"
    fake_root.mkdir()
    venv_file = fake_root / ".venv" / "lib" / "something.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.touch()

    assert not is_project_local(venv_file, fake_root), ".venv path should not be project-local"


@sc.timer()
def test_project_local_outside(tmp_path):
    project_root = tmp_path / "myproject"
    project_root.mkdir()
    outside_file = tmp_path / "other_project" / "mod.py"
    outside_file.parent.mkdir()
    outside_file.touch()

    assert not is_project_local(outside_file, project_root), "File outside project root should not be project-local"


@sc.timer()
def test_find_project_root_git():
    found = find_project_root(_PROJECT_ROOT / "src" / "parsimmon" / "cache.py")
    assert (found / ".git").exists(), f"Expected .git under {found}, but not found"
    assert found == _PROJECT_ROOT.resolve(), f"Expected project root {_PROJECT_ROOT.resolve()}, got {found}"


@sc.timer()
def test_find_project_root_fallback(tmp_path):
    start_file = tmp_path / "subdir" / "module.py"
    start_file.parent.mkdir()
    start_file.touch()

    found = find_project_root(start_file)
    assert found == start_file.resolve().parent, f"Expected fallback to {start_file.resolve().parent}, got {found}"


@sc.timer()
def test_fn_hash_deterministic():
    def my_fn(x):
        return x * 2

    h1 = hash_function_chain(my_fn)
    h2 = hash_function_chain(my_fn)
    assert h1 == h2, f"Function hash should be deterministic; got {h1!r} != {h2!r}"
    assert len(h1) == 16, f"Function hash should be 16 chars, got {len(h1)}"


@sc.timer()
def test_fn_hash_different_functions(tmp_path):
    # hash_function_chain hashes the full module AST, so two functions from
    # modules with different source produce different hashes
    mod_a = tmp_path / "mod_a.py"
    mod_b = tmp_path / "mod_b.py"
    mod_a.write_text("def run(x):\n    return x * 2\n")
    mod_b.write_text("def run(x):\n    return x + 9999\n")

    import importlib.util

    def _load(path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    fn_a = _load(mod_a).run
    fn_b = _load(mod_b).run

    h_a = hash_function_chain(fn_a)
    h_b = hash_function_chain(fn_b)
    assert h_a != h_b, "Functions from modules with different source should produce different hashes"


def _make_cache(tmp_path, **kwargs):
    cache_dir = tmp_path / "cache"
    return SimFileCache(cache_dir, **kwargs)


@sc.timer()
def test_cache_save_load_roundtrip(tmp_path):
    cache = _make_cache(tmp_path)
    key = "abc123def456abcd"
    data = {"x": np.array([1.0, 2.0, 3.0]), "y": [10, 20], "z": "hello"}

    cache.save(key, data, metadata={"label": "test"})
    loaded = cache.load(key)

    assert loaded["y"] == data["y"], f"List mismatch: {loaded['y']}"
    assert loaded["z"] == data["z"], f"String mismatch: {loaded['z']}"
    assert np.array_equal(loaded["x"], data["x"]), f"Array mismatch: {loaded['x']}"


@sc.timer()
def test_cache_exists(tmp_path):
    cache = _make_cache(tmp_path)
    key = "aabbccddeeff0011"

    assert not cache.exists(key), "Cache should not exist before save"
    cache.save(key, {"val": 42}, metadata={})
    assert cache.exists(key), "Cache should exist after save"


@sc.timer()
def test_cache_index(tmp_path):
    cache = _make_cache(tmp_path)
    key = "1122334455667788"
    meta = {"label": "run_1", "fn_hash": "deadbeef12345678"}

    cache.save(key, [1, 2, 3], metadata=meta)
    entries = cache.index()

    assert len(entries) == 1, f"Expected 1 index entry, got {len(entries)}"
    entry = entries[0]
    assert entry["cache_key"] == key, f"Wrong cache_key: {entry['cache_key']}"
    assert entry["label"] == "run_1", f"Wrong label: {entry['label']}"
    assert entry["fn_hash"] == "deadbeef12345678", f"Wrong fn_hash: {entry['fn_hash']}"
    assert "timestamp" in entry, "Index entry should have a timestamp"


@sc.timer()
def test_cache_add_index_entry(tmp_path):
    cache = _make_cache(tmp_path)
    key = "ffffffffffffffff"
    meta = {"cache_key": key, "label": "dedup_hit", "source": "cross-set"}

    cache.add_index_entry(meta)

    assert not cache.exists(key), "add_index_entry should not write a result file"

    entries = cache.index()
    assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
    assert entries[0]["label"] == "dedup_hit", f"Wrong label: {entries[0]}"


@sc.timer()
def test_cache_keys(tmp_path):
    cache = _make_cache(tmp_path)
    assert cache.keys() == [], "Empty cache should return no keys"

    key1 = "aaaa000011110000"
    key2 = "bbbb111122220000"
    cache.save(key1, {"a": 1}, metadata={})
    cache.save(key2, {"b": 2}, metadata={})

    found = sorted(cache.keys())
    assert found == sorted([key1, key2]), f"Expected {sorted([key1, key2])}, got {found}"


@sc.timer()
def test_cache_delete(tmp_path):
    cache = _make_cache(tmp_path)
    key = "deadbeefcafe0000"
    cache.save(key, {"val": 99}, metadata={"label": "to_delete"})

    assert cache.exists(key), "Should exist before delete"
    assert len(cache.index()) == 1

    cache.delete(key)

    assert not cache.exists(key), "Should not exist after delete"
    remaining = [e for e in cache.index() if e.get("cache_key") == key]
    assert remaining == [], "Index entries for deleted key should be removed"


@sc.timer()
def test_cache_clear(tmp_path):
    cache = _make_cache(tmp_path)
    for i, key in enumerate(["aaaa0000aaaa0000", "bbbb1111bbbb1111", "cccc2222cccc2222"]):
        cache.save(key, {"i": i}, metadata={})

    assert len(cache.keys()) == 3
    assert len(cache.index()) == 3

    cache.clear()

    assert cache.keys() == [], "All keys should be gone after clear"
    assert cache.index() == [], "Index should be empty after clear"


@sc.timer()
def test_cache_idempotent_save(tmp_path):
    saved_calls = []

    def tracking_save(path, obj):
        saved_calls.append(path)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def tracking_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    cache = SimFileCache(tmp_path / "cache", save=tracking_save, load=tracking_load)
    key = "idempotent000000"

    cache.save(key, {"v": 1}, metadata={"run": 1})

    # second save with different data -- file should NOT be overwritten
    cache.save(key, {"v": 99}, metadata={"run": 2})

    result_path = str(tmp_path / "cache" / "results" / f"{key}.pkl")
    result_writes = [p for p in saved_calls if result_path in str(p)]
    assert len(result_writes) == 1, f"Result file should be written exactly once; got {len(result_writes)} writes"

    loaded = cache.load(key)
    assert loaded == {"v": 1}, f"Should load first value, got {loaded}"


@sc.timer()
def test_cache_index_persistence(tmp_path):
    cache_dir = tmp_path / "persistent_cache"
    key = "persist000000000"

    cache1 = SimFileCache(cache_dir)
    cache1.save(key, {"data": "original"}, metadata={"label": "persisted"})

    # new instance at same path should read index from disk
    cache2 = SimFileCache(cache_dir)
    entries = cache2.index()

    assert len(entries) == 1, f"Expected 1 entry from disk, got {len(entries)}"
    assert entries[0]["cache_key"] == key, f"Wrong cache_key: {entries[0]}"
    assert entries[0]["label"] == "persisted", f"Wrong label: {entries[0]}"
    assert cache2.exists(key), "Result file should still exist for new instance"


@sc.timer()
def test_cache_custom_save_load(tmp_path):
    call_log = {"saved": [], "loaded": []}

    def custom_save(path, obj):
        call_log["saved"].append(path)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def custom_load(path):
        call_log["loaded"].append(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    cache = SimFileCache(tmp_path / "custom", save=custom_save, load=custom_load)
    key = "custom0000000000"
    obj = {"custom": True, "nums": [1, 2, 3]}

    cache.save(key, obj, metadata={})
    assert len(call_log["saved"]) >= 1, "custom_save should have been called"

    loaded = cache.load(key)
    assert len(call_log["loaded"]) >= 1, "custom_load should have been called"
    assert loaded == obj, f"Loaded value mismatch: {loaded}"


@sc.timer()
def test_cache_get_fn_hash(tmp_path):
    cache = _make_cache(tmp_path)
    key = "fnhash0000000000"
    fn_hash = "abcd1234efgh5678"

    cache.save(key, {"result": 42}, metadata={"fn_hash": fn_hash, "label": "v1"})

    retrieved = cache.get_fn_hash(key)
    assert retrieved == fn_hash, f"Expected {fn_hash!r}, got {retrieved!r}"

    assert cache.get_fn_hash("nonexistent000000") is None, "Missing key should return None"


@sc.timer()
def test_base_not_implemented():
    base = SimCacheBase()
    dummy_meta = {"cache_key": "x", "label": "test"}

    with pytest.raises(NotImplementedError):
        base.save("key", {}, {})
    with pytest.raises(NotImplementedError):
        base.load("key")
    with pytest.raises(NotImplementedError):
        base.exists("key")
    with pytest.raises(NotImplementedError):
        base.index()
    with pytest.raises(NotImplementedError):
        base.add_index_entry(dummy_meta)
    with pytest.raises(NotImplementedError):
        base.keys()
    with pytest.raises(NotImplementedError):
        base.delete("key")
    with pytest.raises(NotImplementedError):
        base.clear()


if __name__ == "__main__":
    T = sc.timer()

    test_hash_deterministic()
    test_hash_key_order_invariant()
    test_hash_differentiates_values()
    test_hash_differentiates_types()
    test_hash_nested_dicts()
    test_hash_numpy_array()
    test_hash_numpy_scalars()
    test_hash_lists_tuples()
    test_hash_none_and_bool()
    test_cache_key_truncated()
    test_cache_key_deterministic()
    test_project_local_inside()
    test_project_local_site_packages()

    import tempfile

    with tempfile.TemporaryDirectory() as _tmp:
        _tmp = Path(_tmp)
        test_project_local_venv(_tmp / "venv_test")
        test_project_local_outside(_tmp / "outside_test")
        test_find_project_root_fallback(_tmp / "fallback_test")
        test_cache_save_load_roundtrip(_tmp / "roundtrip")
        test_cache_exists(_tmp / "exists")
        test_cache_index(_tmp / "index")
        test_cache_add_index_entry(_tmp / "add_entry")
        test_cache_keys(_tmp / "keys")
        test_cache_delete(_tmp / "delete")
        test_cache_clear(_tmp / "clear")
        test_cache_idempotent_save(_tmp / "idempotent")
        test_cache_index_persistence(_tmp / "persistence")
        test_cache_custom_save_load(_tmp / "custom")
        test_cache_get_fn_hash(_tmp / "fn_hash")

    test_find_project_root_git()
    test_fn_hash_deterministic()
    test_fn_hash_different_functions()
    test_base_not_implemented()

    T.toc()
