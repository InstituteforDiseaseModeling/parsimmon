"""
parsimmon: parameter set management for simulation frameworks.

ParameterSet holds a base dict (shared defaults) and named groups
(override dicts). Iteration yields one update dict per (group x
Cartesian expansion point).

ParameterSetManager provides decorator-based registration and a CLI
entry point with optional content-addressed caching and parallel execution.

Builders must return ``Mapping | ParameterSet`` (never None).
"""

import argparse
import concurrent.futures
import inspect
import itertools
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sciris as sc


def _to_nested_objdict(d):
    out = sc.objdict()
    for k, v in d.items():
        out[k] = _to_nested_objdict(v) if isinstance(v, dict) else v
    return out


def _deep_update(base, updates):
    for key, val in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def _iter_leaves(d, prefix=()):
    for key, val in d.items():
        path = prefix + (key,)
        if isinstance(val, dict):
            yield from _iter_leaves(val, path)
        else:
            yield path, val


def _set_nested(d, path, val):
    for key in path[:-1]:
        d = d.setdefault(key, sc.objdict())
    d[path[-1]] = val


def _get_nested(d, dotted_key):
    for part in dotted_key.split("."):
        if not isinstance(d, dict):
            raise KeyError(f"Cannot navigate into non-dict at '{part}' in '{dotted_key}'")
        if part not in d:
            raise KeyError(f"Key '{part}' not found while resolving '{dotted_key}'")
        d = d[part]
    return d


_UNRESOLVED = object()


def _resolve_source_for_display(sk, root, range_paths, link_path):
    if isinstance(sk, _ParamRange):
        return sk.values
    if isinstance(sk, _ParamLink):
        inner = _resolve_link_for_display(sk, root, range_paths, link_path)
        return inner if isinstance(inner, list) else [inner]
    if "." not in sk:
        parent = ".".join(link_path[:-1])
        dotted = f"{parent}.{sk}" if parent else sk
    else:
        dotted = sk
    sv = _get_nested(root, dotted)
    return sv.values if isinstance(sv, _ParamRange) else [sv]


def _resolve_link_for_display(link, root, range_paths, link_path):
    sources = [_resolve_source_for_display(sk, root, range_paths, link_path) for sk in link.source]
    resolved = [link.fn(*combo) for combo in itertools.product(*sources)]
    return resolved if len(resolved) > 1 else resolved[0]


def _resolve_markers(d, root=None, range_paths=None, prefix=()):
    if root is None:
        root = d
        range_paths = {id(val): path for path, val in _iter_leaves(d) if isinstance(val, _ParamRange)}
    out = sc.objdict()
    for key, val in d.items():
        path = prefix + (key,)
        if isinstance(val, dict):
            out[key] = _resolve_markers(val, root, range_paths, path)
        elif isinstance(val, _ParamRange):
            out[key] = val.values
        elif isinstance(val, _ParamLink):
            out[key] = _resolve_link_for_display(val, root, range_paths, path)
        else:
            out[key] = val
    return out


class _ParamRange:
    """Marker for Cartesian expansion; survives deep-merge as a non-dict leaf.

    Created via ParameterSet.arange(), .linspace(), .logspace(), or .iter().
    """

    def __init__(self, values):
        self.values = list(values)

    def link(self, fn):
        return _ParamLink(self, fn)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"_ParamRange({self.values!r})"


class _ParamLink:
    """Marker for derived parameters, resolved after range expansion."""

    def __init__(self, source, fn):
        self.source = [source] if isinstance(source, (str, _ParamRange, _ParamLink)) else list(source)
        self.fn = fn

    def __repr__(self):
        return f"_ParamLink({self.source!r})"


class ParameterSet:
    """Base defaults plus named groups of overrides with Cartesian expansion.

    Iteration yields one resolved dict per (group x range product).
    Attribute access navigates groups then defaults (``ps.G2.a.b``).
    """

    def __init__(self, base=None):
        self._base = sc.objdict()
        self._groups = sc.objdict()
        if base is not None:
            _deep_update(self._base, _to_nested_objdict(sc.dcp(base)))

    def __getattr__(self, name):
        # prevents infinite recursion during deepcopy when _groups/_base
        # aren't in __dict__ yet
        if name.startswith("_"):
            raise AttributeError(name)
        in_groups = name in self._groups
        in_base = name in self._base
        if in_groups and in_base:
            raise AttributeError(f"'{name}' exists as both group and default key. Deduplicate")
        if in_groups:
            return self._groups[name]
        if in_base:
            return self._base[name]
        raise AttributeError(f"'{type(self).__name__}' has no group or default key '{name}'")

    @staticmethod
    def arange(*args, ndigits=3, **kwargs):
        return _ParamRange(np.round(np.arange(*args, **kwargs), ndigits))

    @staticmethod
    def linspace(*args, ndigits=3, **kwargs):
        return _ParamRange(np.round(np.linspace(*args, **kwargs), ndigits))

    @staticmethod
    def logspace(*args, ndigits=3, **kwargs):
        return _ParamRange(np.round(np.logspace(*args, **kwargs), ndigits))

    @staticmethod
    def iter(iterable):
        return _ParamRange(iterable)

    @staticmethod
    def link(source, fn):
        # bare strings resolve as siblings; dotted strings are full paths
        return _ParamLink(source, fn)

    def add(self, name_or_overrides, overrides=None):
        if isinstance(name_or_overrides, str):
            name = name_or_overrides
            if hasattr(ParameterSet, name) or name.startswith("_"):
                raise ValueError(f"Group name '{name}' collides with a ParameterSet attribute")
            if name in self._base:
                raise ValueError(f"Group name '{name}' collides with default key '{name}'")
            ovr = _to_nested_objdict(sc.dcp(overrides)) if overrides is not None else sc.objdict()
            if name in self._groups:
                _deep_update(self._groups[name], ovr)
            else:
                self._groups[name] = ovr
        elif isinstance(name_or_overrides, dict):
            for key in name_or_overrides:
                if key in self._groups:
                    raise ValueError(f"Default key '{key}' collides with group name '{key}'")
            _deep_update(self._base, _to_nested_objdict(sc.dcp(name_or_overrides)))
        else:
            raise TypeError(f"Expected str or dict, got {type(name_or_overrides).__name__}")

    def clear(self, name_or_keys, keys=None):
        if isinstance(name_or_keys, str):
            target = self._groups.get(name_or_keys, sc.objdict())
            key_list = keys or []
        elif isinstance(name_or_keys, (list, tuple)):
            target = self._base
            key_list = name_or_keys
        else:
            raise TypeError(f"Expected str or list, got {type(name_or_keys).__name__}")
        for dotted in key_list:
            path = tuple(dotted.split("."))
            node = target
            for key in path[:-1]:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, dict):
                node.pop(path[-1], None)

    @property
    def groups(self):
        return list(self._groups.keys())

    def __iter__(self):
        sim_id = 1
        for name, overrides in self._groups.items():
            merged = _deep_update(sc.dcp(self._base), sc.dcp(overrides))
            for group_id, (_, updates) in enumerate(self._expand(merged), start=1):
                yield sc.objdict(group=name, sim_id=sim_id, group_id=group_id, pars=updates)
                sim_id += 1

    def __len__(self):
        return sum(self._count(_deep_update(sc.dcp(self._base), sc.dcp(ovr))) for ovr in self._groups.values())

    @staticmethod
    def _expand(merged):
        fixed = {}
        ranges = []
        links = []
        range_paths = {}

        for path, val in _iter_leaves(merged):
            if isinstance(val, _ParamRange):
                ranges.append((path, val))
                range_paths[id(val)] = path
            elif isinstance(val, _ParamLink):
                links.append((path, val))
            else:
                fixed[path] = val

        def _resolve_one(link_obj, updates, link_path):
            source_vals = []
            for sk in link_obj.source:
                if isinstance(sk, _ParamRange):
                    rpath = range_paths.get(id(sk))
                    if rpath is None:
                        raise ValueError(
                            "Linked _ParamRange not found in this group — "
                            "after ps.add('G', ps.Other), link via ps.G.x.y, not ps.Other.x.y"
                        )
                    dotted = ".".join(rpath)
                    source_vals.append(_get_nested(updates, dotted))

                elif isinstance(sk, _ParamLink):
                    for lp, lk in links:
                        if lk is sk:
                            try:
                                source_vals.append(_get_nested(updates, ".".join(lp)))
                            except KeyError:
                                return _UNRESOLVED
                            break
                    else:
                        raise ValueError("Inner _ParamLink not found in link list")

                elif isinstance(sk, str):
                    if "." not in sk:
                        parent = ".".join(link_path[:-1])
                        dotted = f"{parent}.{sk}" if parent else sk
                    else:
                        dotted = sk
                    try:
                        source_vals.append(_get_nested(updates, dotted))
                    except KeyError:
                        return _UNRESOLVED

                else:
                    raise TypeError(f"Unsupported link source type: {type(sk).__name__}")
            return link_obj.fn(*source_vals)

        def _resolve_links(updates):
            unresolved = list(links)
            for _ in range(len(links) + 1):
                still_unresolved = []
                for path, link_obj in unresolved:
                    val = _resolve_one(link_obj, updates, path)
                    if val is _UNRESOLVED:
                        still_unresolved.append((path, link_obj))
                    else:
                        _set_nested(updates, path, val)
                if not still_unresolved:
                    return
                if len(still_unresolved) == len(unresolved):
                    paths = [".".join(p) for p, _ in still_unresolved]
                    raise ValueError(f"Circular link dependency among: {paths}")
                unresolved = still_unresolved

        def _build_and_resolve(point):
            updates = sc.objdict()
            for path, val in fixed.items():
                _set_nested(updates, path, val)
            for path, val in point.items():
                _set_nested(updates, path, val)
            _resolve_links(updates)
            return updates

        if not ranges:
            yield {}, _build_and_resolve({})
            return

        axes = [[{path: v} for v in pr.values] for path, pr in ranges]

        for combo in itertools.product(*axes):
            point = {k: v for d in combo for k, v in d.items()}
            yield point, _build_and_resolve(point)

    @staticmethod
    def _count(merged):
        sizes = [len(val) for _, val in _iter_leaves(merged) if isinstance(val, _ParamRange)]
        return math.prod(sizes) if sizes else 1

    def print_summary(self):
        for name, overrides in self._groups.items():
            merged = _deep_update(sc.dcp(self._base), sc.dcp(overrides))
            resolved = _resolve_markers(merged)
            print(f"\n--- {name} ---")
            sc.pp(resolved)
        print(f"\nTotal: {len(self)} simulations across {len(self._groups)} groups")


class _Entry(NamedTuple):
    builder: callable
    parent_name: str | None = None


class ParameterSetManager:
    """Registry of named parameter-set builders with decorator API and CLI.

    Builders return ``Mapping`` (updates defaults) or ``ParameterSet``
    (used directly). With ``extends``, the child receives a deep copy of
    the fully-built parent.
    """

    def __init__(self, path=None, cache=None, data_dir=None, plots_dir=None):
        self._entries = sc.objdict()
        self._analyses = {}
        self._extra_args = []
        self._default_name = None
        self._path = path
        self._data_dir_override = Path(data_dir) if data_dir is not None else None
        self._plots_dir_override = Path(plots_dir) if plots_dir is not None else None

        if cache is True:
            from .cache import SimFileCache

            self._cache = SimFileCache(self.data_dir / "cache")
        elif cache:
            self._cache = cache
        else:
            self._cache = None

    @property
    def data_dir(self):
        if self._data_dir_override is not None:
            return self._data_dir_override
        base = Path("data")
        return base / self._path if self._path else base

    @property
    def plots_dir(self):
        if self._plots_dir_override is not None:
            return self._plots_dir_override
        base = Path("plots")
        return base / self._path if self._path else base

    def add_argument(self, *args, **kwargs):
        self._extra_args.append((args, kwargs))

    @property
    def _default(self):
        # explicit default=True wins, else last registered
        if self._default_name is not None:
            return self._default_name
        return list(self._entries.keys())[-1] if self._entries else None

    def add(self, fn_or_name=None, extends=None, default=False):
        parent_name = self._fn_to_name(extends) if callable(extends) else extends

        if callable(fn_or_name):
            return self._register(fn_or_name, fn_or_name.__name__, parent_name, default)

        name = fn_or_name  # None or str

        def decorator(fn):
            return self._register(fn, name or fn.__name__, parent_name, default)

        return decorator

    def _register(self, fn, name, parent_name=None, default=False):
        self._entries[name] = _Entry(fn, parent_name)
        fn._pm_name = name
        if default:
            self._default_name = name
        return fn

    def _fn_to_name(self, fn):
        name = getattr(fn, "_pm_name", None)
        if name is not None and name in self._entries:
            return name
        raise KeyError(f"Function {fn.__name__!r} is not registered")

    def analysis(self, name):
        def decorator(fn):
            self._analyses[name] = fn
            return fn

        return decorator

    def _build(self, name, cli_overrides=None):
        entry = self._entries[name]

        if entry.parent_name:
            ps = sc.dcp(self._build(entry.parent_name))
        else:
            ps = ParameterSet()

        sig = inspect.signature(entry.builder)
        needs_ps = any(p.default is p.empty for p in sig.parameters.values())
        result = entry.builder(ps) if needs_ps else entry.builder()

        if isinstance(result, Mapping):
            _deep_update(ps._base, _to_nested_objdict(sc.dcp(result)))
        elif isinstance(result, ParameterSet):
            ps = result
        else:
            raise TypeError(f"Builder must return Mapping or ParameterSet, got {type(result).__name__}")

        if cli_overrides:
            self._apply_cli_overrides(ps, cli_overrides)
        return ps

    @staticmethod
    def _apply_cli_overrides(ps, overrides):
        for raw in overrides:
            if "=" not in raw:
                raise ValueError(f"Invalid override {raw!r}; expected key=value")
            key, val_str = raw.split("=", 1)
            path = tuple(key.split("."))
            try:
                existing = _get_nested(ps._base, key)
                val = _coerce(val_str, type(existing))
            except (KeyError, ValueError):
                val = _auto_coerce(val_str)

            _set_nested(ps._base, path, val)

    def run(self, fn=None, jobs=None, argv=None):
        parser = argparse.ArgumentParser()
        default = self._default
        p_kwargs = {"choices": list(self._entries.keys()), "help": "Parameter set to run"}
        if default is not None:
            p_kwargs["default"] = default
        else:
            p_kwargs["required"] = True
        parser.add_argument("-p", "--parameter-set", **p_kwargs)
        parser.add_argument(
            "-a",
            "--args",
            action="append",
            default=[],
            help="Override key=value (repeatable)",
        )
        parser.add_argument(
            "--print",
            action="store_true",
            dest="print_pars",
            help="Print parameter dicts and exit",
        )
        parser.add_argument("--count", action="store_true", help="Print sim count and exit")
        parser.add_argument(
            "--list",
            action="store_true",
            help="List registered parameter sets and exit",
        )
        parser.add_argument("--no-plot", action="store_true", help="Skip post-run analysis/plotting")

        for extra_args, extra_kwargs in self._extra_args:
            parser.add_argument(*extra_args, **extra_kwargs)

        args = parser.parse_args(argv)
        self.args = args

        if args.list:
            for entry_name in self._entries:
                count = len(self._build(entry_name))
                star = "*" if entry_name == default else " "
                print(f"{star}{entry_name} {count}")
            return None

        name = args.parameter_set
        ps = self._build(name, cli_overrides=args.args or None)

        if args.count:
            print(len(ps))
            return None

        if args.print_pars:
            ps.print_summary()
            return None

        if fn is None:
            raise ValueError("fn (simulation function) is required to run simulations")

        return self._execute(name, ps, fn, jobs, run_analysis=not args.no_plot)

    def _execute(self, name, ps, fn, jobs=None, run_analysis=True):
        from .results import SimResult, _SimEntry

        if jobs is not None and jobs < 1:
            raise ValueError(f"jobs must be a positive integer, got {jobs}")

        sim_points = list(ps)

        fn_hash = None
        if self._cache is not None:
            from .cache import compute_cache_key, hash_function_chain

            fn_hash = hash_function_chain(fn)

        entries = []
        to_run = []

        for i, meta in enumerate(sim_points):
            fn_metadata = {"parameter_set": name, "group": meta.group, "sim_id": meta.sim_id, "group_id": meta.group_id}
            entry_metadata = {"pars": sc.dcp(meta.pars), "fn_hash": fn_hash} | fn_metadata
            cache_key = None

            if self._cache is not None:
                cache_key = compute_cache_key(meta.pars)
                if self._cache.exists(cache_key):
                    stored_fn_hash = self._cache.get_fn_hash(cache_key)
                    if stored_fn_hash is not None and fn_hash != stored_fn_hash:
                        warnings.warn(
                            f"Simulation function has changed since the result "
                            f"for key {cache_key} was cached. Using cached result.",
                            stacklevel=3,
                        )
                    entry_metadata["cache_key"] = cache_key
                    self._cache.add_index_entry(entry_metadata)
                    entries.append(
                        _SimEntry(
                            metadata=entry_metadata,
                            cache_key=cache_key,
                            backend=self._cache,
                        )
                    )
                    continue

            to_run.append((i, meta.pars, fn_metadata, entry_metadata, cache_key))
            entries.append(None)

        if to_run:
            n_cached = len(sim_points) - len(to_run)
            if n_cached > 0:
                print(f"Running {len(to_run)} simulations ({n_cached} cached)...")
            else:
                print(f"Running {len(to_run)} simulations...")

            if jobs is not None and len(to_run) > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                    future_map = {}
                    for run_idx, (_, pars, fn_meta, _, _) in enumerate(to_run):
                        future = executor.submit(fn, pars, fn_meta)
                        future_map[future] = run_idx
                    ordered_results = [None] * len(to_run)
                    for future in concurrent.futures.as_completed(future_map):
                        ordered_results[future_map[future]] = future.result()
            else:
                ordered_results = [fn(pars, fn_meta) for _, pars, fn_meta, _, _ in to_run]

            for (entry_idx, _, _, entry_meta, cache_key), result in zip(to_run, ordered_results):
                if self._cache is not None and cache_key is not None:
                    self._cache.save(cache_key, result, entry_meta)
                    entry_meta["cache_key"] = cache_key
                    entries[entry_idx] = _SimEntry(
                        metadata=entry_meta,
                        cache_key=cache_key,
                        backend=self._cache,
                    )
                else:
                    entries[entry_idx] = _SimEntry(metadata=entry_meta, value=result)
        else:
            print(f"All {len(sim_points)} simulations loaded from cache.")

        sim_result = SimResult(entries)

        if run_analysis and name in self._analyses:
            print(f"Running analysis for '{name}'...")
            self._analyses[name](sim_result)

        return sim_result

    def results(self):
        from .results import SimResult

        if self._cache is None:
            raise RuntimeError("Caching is not enabled; pass cache=True to ParameterSetManager")
        return SimResult.from_cache(self._cache)


def _coerce(val_str, target_type):
    if target_type is bool:
        if val_str.lower() in {"true", "1", "yes"}:
            return True
        if val_str.lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot coerce {val_str!r} to bool")
    return target_type(val_str)


def _auto_coerce(val_str):
    for converter in (int, float):
        try:
            return converter(val_str)
        except (ValueError, TypeError):
            pass
    if val_str.lower() in {"true", "false"}:
        return val_str.lower() == "true"
    return val_str
