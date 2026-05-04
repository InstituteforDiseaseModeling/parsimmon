# parsimmon

> **WIP** -- This library is under active development. APIs may change.

**Par**ameter and **Sim**ulation **Mon**agement

A **low-friction** library for defining, sweeping, and caching simulation parameter sets in Python.

<details>
  <summary>LLM-guided setup for your project</summary>

  For LLM-guided setup in your simulation project, copy this to your coding agent:

  ```
  Fetch and follow the instructions at https://github.com/InstituteforDiseaseModeling/parsimmon/blob/main/llm_install.md
  ```
</details>

## Install

```bash
# From GitHub directly
pip install git+https://github.com/InstituteforDiseaseModeling/parsimmon.git

# Or clone and install locally
git clone https://github.com/InstituteforDiseaseModeling/parsimmon.git
pip install ./parsimmon
```

## Quick Start

Define your parameters in one file, your simulation in another.

**`parameters.py`**

```python
from parsimmon import ParameterSet, ParameterSetManager

pm = ParameterSetManager()

# Base parameter set -- just return a dict.
@pm.add
def base():
    return {
        'beta': 0.3,
        'n': 1000,
        'seed': ParameterSet.arange(0, 5, 1),
    }

# Extend base with experimental groups.
@pm.add(extends=base)
def experiment(ps):
    ps.add('baseline', {'beta': 0.3})
    ps.add('treatment', {'beta': 0.1})
    return ps
```

**`run_sim.py`**

```python
from parameters import pm

def run(pars, metadata):
    return {'infected': pars.n * pars.beta}

if __name__ == '__main__':
    results = pm.run(run)
```

Run from the command line:

```bash
python run_sim.py                   # runs last registered parameter set
python run_sim.py base              # run a specific parameter set
python run_sim.py --print           # print all parameter summaries and exit
python run_sim.py --count           # print total sim count and exit
```


## Core Concepts

### ParameterSet

A `ParameterSet` holds a dict of base defaults and zero or more named groups of
overrides. When iterated, it yields one resolved parameter dict per combination
of (group, Cartesian expansion point).

**Defaults and groups**

```python
ps = ParameterSet()
ps.add({'beta': 0.3, 'n': 1000})     # set base defaults
ps.add('control', {'beta': 0.1})     # named group overrides beta
ps.add('treated', {'beta': 0.5})
```

**Ranges** -- Cartesian sweep over one or more axes:

```python
ps.add('sweep', {
    'beta': ParameterSet.arange(0.1, 0.5, 0.1),   # [0.1, 0.2, 0.3, 0.4]
    'n':    ParameterSet.linspace(500, 2000, 4),   # 4 evenly spaced values
    'days': ParameterSet.iter([30, 60, 90]),        # explicit iterable
})
# yields 4 * 4 * 3 = 48 parameter dicts for group 'sweep'
```

Ranges can also be created with `ParameterSet.logspace(start, stop, num)`.

**Links** -- derive a parameter value from a range after expansion:

```python
beta_range = ParameterSet.arange(0.1, 0.5, 0.1)
ps.add('sweep', {
    'beta':   beta_range,
    'r0':     beta_range.link(lambda b: b * 10),   # tracks beta exactly
    'budget': ParameterSet.link('n', lambda n: n * 50),  # from sibling key
})
```

Link sources can be a `_ParamRange` object (identity reference), a dotted key
string resolved from the root (`'a.b.c'`), a bare key string resolved as a
sibling, another `_ParamLink` (chain), or a list mixing these. Circular
dependencies raise `ValueError`.

**Attribute access**

```python
ps.control          # returns the group override dict for 'control'
ps.beta             # returns the base default value for 'beta'
ps.print_summary()  # pretty-print all groups with ranges resolved
len(ps)             # total number of simulations across all groups
```

**Label modes** -- control how simulation labels are generated:

| `label=`      | Behaviour |
|---------------|-----------|
| `'enumerate'` | `'{group}{global_sim_id}'` (default) |
| `'zip'`       | `'{group}{group_local_id}'` |
| `None`        | bare group name |
| callable      | `fn(group_name, sim_id, updates)` |


### ParameterSetManager

`ParameterSetManager` is a registry of named parameter-set builders. It provides
a decorator API and a CLI entry point.

**Registration**

```python
pm = ParameterSetManager()

@pm.add                          # name from function name
def base():
    return {'beta': 0.3}

@pm.add(extends=base)            # child receives deep copy of parent
def extended(ps):
    ps.add('A', {'beta': 0.1})
    ps.add('B', {'beta': 0.8})
    return ps

@pm.add(default=True)           # explicitly mark as CLI default
def main(ps):
    ...
    return ps
```

**Analysis**

```python
@pm.analysis('experiment')
def plot_experiment(results):
    # called automatically after pm.run() unless --skip-analysis is passed
    for r in results:
        print(r)
```

**Running**

```python
results = pm.run(fn)             # sequential
results = pm.run(fn, jobs=4)     # parallel via ProcessPoolExecutor
```

`pm.run()` parses `sys.argv`. Useful CLI flags:

| Flag | Effect |
|------|--------|
| `NAME` | Positional: select which registered set to run |
| `-a key=value [...]` | Override base parameters (multiple allowed) |
| `--print` | Print parameter summaries and exit (one if NAME given, else all) |
| `--count` | Print simulation count and exit |
| `--list` | List parameter sets and exit (details if NAME given, else all) |
| `-s` / `--skip-analysis` | Skip post-run analysis |

Extra CLI arguments can be registered with `pm.add_argument(*args, **kwargs)`,
which forwards directly to `argparse`.

**Directory layout defaults**

| Property | Default path |
|----------|--------------|
| `pm.data_dir` | `data/` or `data/{path}` |
| `pm.plots_dir` | `plots/` or `plots/{path}` |

Both can be overridden via constructor parameters `data_dir=` and `plots_dir=`.


### SimResult

`SimResult[T]` is the lazy, navigable container returned by `pm.run()`. Results
are not loaded from disk until accessed by integer index or iteration.

**Iteration and indexing**

```python
for result in results:          # loads each result in turn
    print(result)

results[0]                      # loads the first result
results[1:5]                    # slice returns a new SimResult (no loading)
len(results)                    # count without loading
```

**Attribute navigation** -- navigate to a subset by parameter set or group name:

```python
results.experiment              # SimResult containing only 'experiment' entries
results.experiment.baseline     # further filtered to group 'baseline'
```

**Grouping**

```python
results.groups                  # sc.objdict mapping group name -> SimResult
results.group('beta')           # partition by a metadata/pars key value
results.group('sim.label')      # dotted key path supported
```

**Filtering**

```python
# by dotted key + value
high = results.filter('beta', 0.5)

# by predicate receiving (pars, metadata)
fast = results.filter(lambda pars, meta: pars.get('beta', 0) > 0.3)
```

**Metadata access** (never triggers loading)

```python
results.pars        # list of parameter dicts for all entries
results.metadata    # list of metadata dicts for all entries
```

Each metadata dict contains: `parameter_set`, `group`, `sim_id`, `group_id`,
`label`, `pars`, `fn_hash`, and (if cached) `cache_key` and `timestamp`.


## Caching

Caching is opt-in. Pass `cache=True` to enable the default file-system cache:

```python
pm = ParameterSetManager(cache=True)
```

The cache is stored under `pm.data_dir / 'cache'` by default.

**How it works**

- The cache key is the first 16 hex characters of the SHA-256 of a canonical
  parameter representation. Two parameter dicts with identical values (including
  nested dicts, numpy arrays, and scalars) produce the same key regardless of
  insertion order.
- A `fn_hash` is computed from the AST of the simulation function and all
  project-local modules it imports. This hash is stored as metadata alongside
  each cached result.
- If the simulation function changes between runs, parsimmon emits a warning but
  still returns the cached result (lenient mode). This avoids accidental
  re-computation while keeping you informed of potential staleness.
- Two parameter sets that resolve to the same parameter dict share a single
  result file on disk (cross-set deduplication).

**Cache directory layout**

```
data/cache/
    index.cache         -- serialized list of all metadata dicts
    results/
        {cache_key}.pkl -- one file per unique parameter combination
```

The index is updated atomically (write-then-rename) so a crash during a run
cannot leave a corrupt index.

**Loading all cached results**

```python
results = pm.results()   # SimResult built from the full cache index
```


## Customizing Save/Load

`SimFileCache` accepts optional `save` and `load` callables, allowing you to
control the serialization format for both result files and the index:

```python
import json

def my_save(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)

def my_load(path):
    with open(path) as f:
        return json.load(f)

pm = ParameterSetManager(
    cache=SimFileCache('data/cache', save=my_save, load=my_load)
)
```

The `save` callable has signature `(path: str, obj: Any) -> None` and `load`
has signature `(path: str) -> Any`. Both callables are used for result files
and for the index, so the serialization format must support arbitrary Python
objects (or you must ensure your results are JSON-serialisable, as in the
example above).

The default serializer is `sciris.save` / `sciris.load` (pickle-based), falling
back to the stdlib `pickle` module if sciris is not installed.


## Custom Cache Backends

Subclass `SimCacheBase` to implement alternative storage (SQLite, a remote
object store, a database, etc.). The backend is responsible for both storing
results and maintaining the index.

```python
from parsimmon import SimCacheBase

class MyBackend(SimCacheBase):
    def __init__(self, connection_string):
        self._conn = ...

    def save(self, cache_key, result, metadata):
        # persist result and append to index
        ...

    def load(self, cache_key):
        # return the result object
        ...

    def exists(self, cache_key):
        # return True if a result is stored for cache_key
        ...

    def index(self):
        # return list of metadata dicts for all stored results
        ...

    def add_index_entry(self, metadata):
        # append a metadata entry without writing a result file
        # (used for cross-set deduplication hits)
        ...

    def keys(self):
        # return list of all cache keys currently stored
        ...

    def delete(self, cache_key):
        ...

    def clear(self):
        ...

pm = ParameterSetManager(cache=MyBackend('...'))
```
