"""
Lazy, navigable collection of simulation results.

SimResult[T] replaces SimGroup as the primary result container. It holds
entries that may be backed by an in-memory value or a cache reference; actual
result objects are loaded only when iterated or indexed by integer. Grouping,
filtering, and attribute navigation never trigger loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Iterator, TypeVar

import sciris as sc

T = TypeVar("T")

# sentinel distinguishing "not yet loaded" from None (a valid result value)
_UNLOADED = object()


def _get_nested(d, dotted_key):
    for part in dotted_key.split("."):
        if not isinstance(d, dict):
            raise KeyError(f"Cannot navigate into non-dict at '{part}' in '{dotted_key}'")
        if part not in d:
            raise KeyError(f"Key '{part}' not found while resolving '{dotted_key}'")
        d = d[part]
    return d


def _get_meta_value(metadata, key):
    # checks top-level keys first, then falls into metadata.pars
    first = key.split(".")[0]
    if first in metadata:
        return _get_nested(metadata, key)
    return _get_nested(metadata.pars, key)


class _SimEntry:
    __slots__ = ("metadata", "_value", "_cache_key", "_backend")

    def __init__(self, metadata, value=None, cache_key=None, backend=None):
        self.metadata = metadata if isinstance(metadata, sc.objdict) else sc.objdict(metadata)
        self._value = _UNLOADED if value is None and cache_key is not None else value
        self._cache_key = cache_key
        self._backend = backend

    def resolve(self) -> Any:
        if self._value is _UNLOADED:
            if self._backend is None or self._cache_key is None:
                raise RuntimeError("No cached result and no in-memory value")
            self._value = self._backend.load(self._cache_key)
        return self._value


class SimResult(Generic[T]):
    """Lazy, navigable collection of simulation results.

    Entries hold either an in-memory value or a (cache_key, backend) pair;
    actual results load only on iteration or integer indexing. Grouping,
    filtering, and metadata access never trigger loading.
    """

    def __init__(self, entries: list):
        self._entries = list(entries)

    @classmethod
    def from_entries(cls, entries) -> "SimResult[T]":
        return cls(entries)

    @classmethod
    def load(cls, path_or_backend) -> "SimResult":
        return cls.from_cache(path_or_backend)

    @classmethod
    def from_cache(cls, path_or_backend) -> "SimResult":
        from .cache import SimFileCache, SimCacheBase  # type: ignore[import]

        if isinstance(path_or_backend, (str, Path)):
            backend = SimFileCache(Path(path_or_backend))
        elif isinstance(path_or_backend, SimCacheBase):
            backend = path_or_backend
        else:
            raise TypeError(f"Expected str, Path, or SimCacheBase, got {type(path_or_backend).__name__}")

        index = backend.index()
        entries = [_SimEntry(metadata=entry, cache_key=entry["cache_key"], backend=backend) for entry in index]
        return cls(entries)

    @classmethod
    def from_values(cls, values, metadata_list) -> "SimResult":
        if len(values) != len(metadata_list):
            raise ValueError(
                f"values and metadata_list must have the same length ({len(values)} vs {len(metadata_list)})"
            )
        entries = [_SimEntry(metadata=meta, value=val) for val, meta in zip(values, metadata_list)]
        return cls(entries)

    def __iter__(self) -> Iterator[T]:
        for entry in self._entries:
            yield entry.resolve()

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._entries[key].resolve()
        if isinstance(key, slice):
            # slicing never loads results -- returns a lightweight view
            return SimResult(self._entries[key])
        raise TypeError(f"Indices must be integers or slices, not {type(key).__name__}")

    def __bool__(self) -> bool:
        return len(self._entries) > 0

    def __getattr__(self, name: str):
        # private/dunder names skip group lookup to prevent infinite recursion
        # during __init__ and deepcopy before _entries exists
        if name.startswith("_") or name in type(self).__dict__:
            raise AttributeError(name)

        parameter_sets = {e.metadata.parameter_set for e in self._entries}
        groups = {e.metadata.group for e in self._entries}

        # if entries span multiple parameter sets, resolve by parameter_set first
        if len(parameter_sets) > 1 and name in parameter_sets:
            filtered = [e for e in self._entries if e.metadata.parameter_set == name]
            return SimResult(filtered)

        if name in groups:
            filtered = [e for e in self._entries if e.metadata.group == name]
            return SimResult(filtered)

        available = sorted(parameter_sets | groups)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'; available parameter sets / groups: {available}"
        )

    @property
    def groups(self):
        buckets = {}
        for entry in self._entries:
            key = entry.metadata.group
            buckets.setdefault(key, []).append(entry)
        result = {k: SimResult(v) for k, v in buckets.items()}
        return sc.objdict(result)

    def group(self, key: str):
        buckets = {}
        for entry in self._entries:
            val = _get_meta_value(entry.metadata, key)
            buckets.setdefault(val, []).append(entry)
        return {k: SimResult(v) for k, v in buckets.items()}

    def filter(self, key_or_fn, value=None) -> "SimResult[T]":
        if callable(key_or_fn):
            filtered = [e for e in self._entries if key_or_fn(e.metadata.pars, e.metadata)]
        else:
            filtered = [e for e in self._entries if _get_meta_value(e.metadata, key_or_fn) == value]
        return SimResult(filtered)

    @property
    def pars(self):
        return [e.metadata.pars for e in self._entries]

    @property
    def metadata(self):
        return [e.metadata for e in self._entries]

    def __repr__(self) -> str:
        n = len(self._entries)
        group_names = list(dict.fromkeys(e.metadata.group for e in self._entries))
        if group_names:
            return f"SimResult(n={n}, groups={group_names!r})"
        return f"SimResult(n={n})"
