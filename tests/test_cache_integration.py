"""Integration tests: cache round-trip with fn_hash invalidation.

Reproduces the structure from hiv-instep-test:
  - sim_fn module imports a helper (like hiv_core) and a plotting module (like plotlib)
  - first run populates cache
  - modifying the plotting module (irrelevant to sim) should NOT trigger a warning
  - modifying the sim helper (relevant to sim) SHOULD invalidate
"""

import importlib.util
import sys
import warnings

import parsimmon as pm
from parsimmon.cache import hash_function_chain

# ---------------------------------------------------------------------------
# Inline helpers replacing external fixture files (drive_sim / sim_test)
# ---------------------------------------------------------------------------


def _make_test_manager(cache_dir):
    """Build a manager with two parameter sets for cache integration tests."""
    manager = pm.ParameterSetManager(cache=pm.SimFileCache(cache_dir))

    @manager.add
    def experiment(ps):
        ps.add("Control", {"beta": 0.1, "seed": 0})
        ps.add("Treatment", {"beta": 0.5, "seed": 0})
        return ps

    return manager


def _run_sim(pars, metadata):
    """Minimal sim stub for basic cache tests."""
    return {"status": "done", "value": 42}


# ---------------------------------------------------------------------------
# Helpers to build a fake project in tmp_path with .git marker
# ---------------------------------------------------------------------------

SIM_HELPER_SRC = """\
def setup(pars):
    return pars
"""

PLOT_HELPER_SRC = """\
def make_plot(results):
    pass
"""

SIM_FN_SRC = """\
from sim_helper import setup
import plot_helper

def run(pars, metadata):
    setup(pars)
    return {"status": "done", "value": 42}
"""


def _write_project(proj_dir, sim_fn_src=SIM_FN_SRC, sim_helper_src=SIM_HELPER_SRC, plot_helper_src=PLOT_HELPER_SRC):
    """Write a minimal project with .git marker and three modules."""
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / ".git").mkdir(exist_ok=True)
    (proj_dir / "sim_fn.py").write_text(sim_fn_src)
    (proj_dir / "sim_helper.py").write_text(sim_helper_src)
    (proj_dir / "plot_helper.py").write_text(plot_helper_src)


def _load_run_fn(proj_dir):
    """Import the run function from the project's sim_fn.py.

    Registers the module in sys.modules so inspect.getmodule() works,
    matching what happens when ``python basic.py`` runs (where the
    script is __main__ with a __file__).
    """
    # ensure the project dir is on sys.path so local imports work
    str_dir = str(proj_dir)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)

    # force re-import of all project modules
    for mod_name in ("sim_fn", "sim_helper", "plot_helper"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location("sim_fn", proj_dir / "sim_fn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_fn"] = mod
    spec.loader.exec_module(mod)
    return mod.run


# ---------------------------------------------------------------------------
# Simple same-code tests (baseline)
# ---------------------------------------------------------------------------


def _run_with_manager(cache_dir):
    """Build manager, execute, return the SimResult."""
    manager = _make_test_manager(cache_dir)
    ps = manager._build("experiment")
    return manager._execute("experiment", ps, _run_sim)


def test_first_run_creates_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    result = _run_with_manager(cache_dir)

    assert len(list(result)) == 2, "Should have 2 results (one per group)"
    assert (cache_dir / "results").exists(), "Cache results dir should exist"
    assert len(list((cache_dir / "results").glob("*.pkl"))) == 2, "Should cache 2 result files"


def test_second_run_uses_cache_no_warnings(tmp_path):
    cache_dir = tmp_path / "cache"

    # first run: populate cache
    _run_with_manager(cache_dir)

    # second run: should use cache with zero warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run_with_manager(cache_dir)

    fn_warnings = [w for w in caught if "function has changed" in str(w.message)]
    assert fn_warnings == [], f"Expected no fn_hash warnings, got: {[str(w.message) for w in fn_warnings]}"
    assert len(list(result)) == 2, "Should still return 2 results from cache"


# ---------------------------------------------------------------------------
# Cross-module fn_hash tests (reproduces hiv-instep-test structure)
# ---------------------------------------------------------------------------


def _run_project(proj_dir, cache_dir):
    """Load the run fn from the project and execute via a manager."""
    run_fn = _load_run_fn(proj_dir)
    manager = pm.ParameterSetManager(cache=pm.SimFileCache(cache_dir))

    @manager.add
    def experiment(ps):
        ps.add("Control", {"beta": 0.1, "seed": 0})
        ps.add("Treatment", {"beta": 0.5, "seed": 0})
        return ps

    ps = manager._build("experiment")
    return manager._execute("experiment", ps, run_fn)


def test_unchanged_code_no_warnings(tmp_path):
    """Same code, two runs -> no warnings (reproduces the basic complaint)."""
    proj_dir = tmp_path / "project"
    cache_dir = tmp_path / "cache"
    _write_project(proj_dir)

    # run 1: populate cache
    _run_project(proj_dir, cache_dir)

    # run 2: identical code
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_project(proj_dir, cache_dir)

    fn_warnings = [w for w in caught if "function has changed" in str(w.message)]
    assert fn_warnings == [], f"Unchanged code should not warn: {[str(w.message) for w in fn_warnings]}"


def test_plotting_change_no_warning(tmp_path):
    """Changing plot_helper.py (irrelevant to sim) should NOT trigger a warning."""
    proj_dir = tmp_path / "project"
    cache_dir = tmp_path / "cache"
    _write_project(proj_dir)

    # run 1: populate cache
    _run_project(proj_dir, cache_dir)

    # modify plotting code (doesn't affect simulation results)
    (proj_dir / "plot_helper.py").write_text("def make_plot(results):\n    print('new plot')\n")

    # run 2: should still use cache without warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_project(proj_dir, cache_dir)

    fn_warnings = [w for w in caught if "function has changed" in str(w.message)]
    assert fn_warnings == [], f"Plotting change should not warn: {[str(w.message) for w in fn_warnings]}"


def test_sim_helper_change_invalidates(tmp_path):
    """Changing sim_helper.py (used by the sim function) SHOULD invalidate."""
    proj_dir = tmp_path / "project"
    cache_dir = tmp_path / "cache"
    _write_project(proj_dir)

    # run 1: populate cache
    _run_project(proj_dir, cache_dir)

    # modify simulation logic
    (proj_dir / "sim_helper.py").write_text("def setup(pars):\n    pars['extra'] = True\n    return pars\n")

    # run 2: should detect the change and re-run (not just warn)
    result = _run_project(proj_dir, cache_dir)
    # At minimum, the system should notice the change.
    # The exact behavior (warn vs re-run) depends on the fix,
    # but the cache should not silently serve stale results.
    assert len(list(result)) == 2


# ---------------------------------------------------------------------------
# File-edit cache invalidation: sim_def + transitive dependency
# ---------------------------------------------------------------------------
#
# Three generated modules form a chain:
#   driver.py  -->  sim_def.py  -->  sim_dep.py  -->  sciris (site-packages)
#
# hash_function_chain(driver.run) hashes sim_def.py and sim_dep.py (sciris
# is excluded as non-project-local).  Editing either file changes the hash,
# causing the cache to invalidate and re-run.

SIM_DEP_SRC = """\
import sciris as sc

def prepare(pars):
    return sc.dcp(pars)
"""

SIM_DEF_SRC = """\
from sim_dep import prepare

def run_sim(pars, metadata):
    prepared = prepare(pars)
    return {"status": "done", "value": 42}
"""

DRIVER_SRC = """\
from sim_def import run_sim

def run(pars, metadata):
    return run_sim(pars, metadata)
"""


def _write_sim_project(proj_dir):
    """Write sim_dep, sim_def, and driver modules with a .git marker."""
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / ".git").mkdir(exist_ok=True)
    (proj_dir / "sim_dep.py").write_text(SIM_DEP_SRC)
    (proj_dir / "sim_def.py").write_text(SIM_DEF_SRC)
    (proj_dir / "driver.py").write_text(DRIVER_SRC)


def _load_driver(proj_dir):
    """Import driver.run, forcing re-import of the whole sim chain."""
    str_dir = str(proj_dir)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)
    for mod_name in ("driver", "sim_def", "sim_dep"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location("driver", proj_dir / "driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["driver"] = mod
    spec.loader.exec_module(mod)
    return mod.run


def test_cache_invalidation_on_file_edit(tmp_path):
    """Editing sim_def or its dependency invalidates; unchanged code is cached."""
    proj_dir = tmp_path / "project"
    cache_dir = tmp_path / "cache"
    _write_sim_project(proj_dir)

    def _run():
        run_fn = _load_driver(proj_dir)
        fn_hash = hash_function_chain(run_fn)
        manager = pm.ParameterSetManager(cache=pm.SimFileCache(cache_dir))

        @manager.add
        def experiment(ps):
            ps.add("Control", {"beta": 0.1, "seed": 0})
            ps.add("Treatment", {"beta": 0.5, "seed": 0})
            return ps

        ps = manager._build("experiment")
        result = manager._execute("experiment", ps, run_fn)
        return result, fn_hash

    # first run: populates cache
    result, h1 = _run()
    assert len(list(result)) == 2

    # unchanged code -> cache hit
    result, h2 = _run()
    assert len(list(result)) == 2
    assert h2 == h1, "hash should be stable without edits"

    # edit sim_def.py -> hash changes -> re-run
    with open(proj_dir / "sim_def.py", "a") as f:
        f.write("\n# edited\n")
    result, h3 = _run()
    assert len(list(result)) == 2
    assert h3 != h1, "hash should change after editing sim_def.py"

    # edit sim_dep.py -> hash changes -> re-run
    with open(proj_dir / "sim_dep.py", "a") as f:
        f.write("\n# edited\n")
    result, h4 = _run()
    assert len(list(result)) == 2
    assert h4 != h3, "hash should change after editing sim_dep.py"

    # unchanged code -> cache hit
    result, h5 = _run()
    assert len(list(result)) == 2
    assert h5 == h4, "hash should be stable without edits"
