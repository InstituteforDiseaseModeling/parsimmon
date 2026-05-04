"""Content-addressed caching for parsimmon simulation runs."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sciris as sc


def _make_default_serializers():
    return (
        lambda path, obj: sc.save(str(path), obj, verbose=False),
        lambda path: sc.load(str(path)),
    )


def _canonical_repr(obj, _seen=None):
    """Build deterministic bytes for hashing; circular references become a
    stable sentinel via the _seen id set rather than recursing infinitely."""
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    # only track mutable containers that can be circular
    is_container = isinstance(obj, (dict, list, tuple))
    if is_container:
        if obj_id in _seen:
            return b"<circular>"
        _seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: repr(kv[0]))
            parts = [b"dict:{"]
            for k, v in items:
                parts.append(_canonical_repr(k, _seen))
                parts.append(b":")
                parts.append(_canonical_repr(v, _seen))
                parts.append(b",")
            parts.append(b"}")
            return b"".join(parts)

        if isinstance(obj, (list, tuple)):
            tag = b"list:[" if isinstance(obj, list) else b"tuple:("
            close = b"]" if isinstance(obj, list) else b")"
            parts = [tag]
            for item in obj:
                parts.append(_canonical_repr(item, _seen))
                parts.append(b",")
            parts.append(close)
            return b"".join(parts)

        if isinstance(obj, np.ndarray):
            header = f"ndarray:{obj.dtype}:{obj.shape}:".encode()
            return header + obj.tobytes()

        # normalize numpy scalars so np.float64(1.0) hashes the same as float(1.0)
        if isinstance(obj, np.integer):
            obj = int(obj)
        elif isinstance(obj, np.floating):
            obj = float(obj)
        elif isinstance(obj, np.bool_):
            obj = bool(obj)

        # bool before int because bool is a subclass of int
        if isinstance(obj, bool):
            return f"bool:{obj!r}".encode()
        if isinstance(obj, int):
            return f"int:{obj!r}".encode()
        if isinstance(obj, float):
            return f"float:{obj!r}".encode()
        if isinstance(obj, str):
            return f"str:{obj!r}".encode()
        if obj is None:
            return b"None"
        if isinstance(obj, bytes):
            return b"bytes:" + obj

        if callable(obj):
            raise TypeError(
                f"Cannot hash callable {obj!r} for caching. "
                f"Parameterize function selection via a bool, number, or name "
                f"and resolve it to the callable in your own code."
            )

        # general fallback: accept any type whose repr is stable across copies
        r = repr(obj)
        if "0x" in r:
            raise TypeError(
                f"Cannot hash object of type {type(obj).__name__} for caching: "
                f"repr contains a memory address, so the cache key would not "
                f"be stable across sessions."
            )
        try:
            obj2 = copy.deepcopy(obj)
        except (TypeError, RecursionError, ValueError):
            raise TypeError(
                f"Cannot hash object of type {type(obj).__name__} for caching: "
                f"deepcopy failed, so repr stability cannot be verified."
            )
        try:
            r2 = repr(obj2)
        except (TypeError, AttributeError, RuntimeError):
            raise TypeError(
                f"Cannot hash object of type {type(obj).__name__} for caching: "
                f"repr failed on the copied object, so repr stability cannot be verified."
            )
        if r != r2:
            raise TypeError(
                f"Cannot hash object of type {type(obj).__name__} for caching: "
                f"repr is not stable across copies ({r!r} != {r2!r})."
            )
        return f"obj:{r}".encode()

    finally:
        if is_container:
            _seen.discard(obj_id)


def hash_params(pars: dict) -> str:
    raw = _canonical_repr(pars)
    return hashlib.sha256(raw).hexdigest()


def compute_cache_key(pars: dict) -> str:
    # 16 hex chars: short enough for filenames, collision-safe for any
    # realistic parameter space
    return hash_params(pars)[:16]


def is_project_local(module_path: Path, project_root: Path) -> bool:
    path_str = str(module_path)
    if "/site-packages/" in path_str or "\\site-packages\\" in path_str:
        return False

    try:
        rel = module_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return False

    venv_markers = {".venv", "venv", "env", ".env", ".tox", ".nox"}
    return not (rel.parts and rel.parts[0] in venv_markers)


def find_project_root(start: Path) -> Path:
    # .git as root marker so the hash boundary matches version control;
    # falls back to parent of start for notebooks / test scripts
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve().parent


def _fn_referenced_names(fn: Callable) -> set[str]:
    """Return the set of global names referenced in *fn*'s body."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return set()

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _resolve_fn_source_files(fn: Callable) -> list[Path]:
    """Find project-local source files that *fn* depends on at runtime.

    Walks *fn*'s AST for referenced names, resolves them through
    ``fn.__globals__`` to real objects, then uses
    ``inspect.getsourcefile`` to locate the file each lives in.
    Returns only project-local files, excluding *fn*'s own module.
    """
    names = _fn_referenced_names(fn)
    fn_file = inspect.getsourcefile(fn)
    fn_globals = getattr(fn, "__globals__", {})

    seeds: list[Path] = []
    for name in names:
        obj = fn_globals.get(name)
        if obj is None:
            continue
        try:
            src_file = inspect.getsourcefile(obj)
        except (TypeError, OSError):
            continue
        if src_file is None:
            continue

        path = Path(src_file)
        path_str = str(path)
        if "/site-packages/" in path_str or "\\site-packages\\" in path_str:
            continue
        # skip fn's own file (the driver)
        if fn_file and Path(fn_file).resolve() == path.resolve():
            continue

        seeds.append(path)
    return seeds


def hash_function_chain(fn: Callable) -> str:
    """Content hash of the project-local files that *fn* depends on.

    Walks *fn*'s body to discover the globals it references, resolves
    them to source files, then transitively follows project-local
    imports.  Hashes raw file contents so any edit (including
    whitespace / comments) invalidates the cache.
    """
    seed_files = _resolve_fn_source_files(fn)

    if not seed_files:
        # no external project-local deps; fall back to fn source
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            src = ""
        return hashlib.sha256(src.encode()).hexdigest()[:16]

    fn_file = inspect.getsourcefile(fn)
    project_root = find_project_root(Path(fn_file)) if fn_file else Path.cwd()

    collected: dict[str, bytes] = {}  # abs_path -> raw content
    for seed in seed_files:
        _collect_local_files(seed, project_root, collected, seen=set())

    combined = b"".join(collected[k] for k in sorted(collected))
    return hashlib.sha256(combined).hexdigest()[:16]


def _collect_local_files(file_path: Path, project_root: Path, collected: dict, seen: set) -> None:
    """Recursively gather raw contents of *file_path* and its project-local imports."""
    abs_path = str(file_path.resolve())
    if abs_path in seen:
        return
    seen.add(abs_path)

    if not is_project_local(file_path, project_root):
        return

    try:
        content = file_path.read_bytes()
    except OSError:
        return

    collected[abs_path] = content

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        dep_path = _resolve_import_node(node, project_root)
        if dep_path is not None:
            _collect_local_files(dep_path, project_root, collected, seen)


def _resolve_import_node(node: ast.AST, project_root: Path) -> "Path | None":
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            return None
        names = [node.module]
    else:
        return None

    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            continue

        if spec is None or spec.origin is None:
            continue

        candidate = Path(spec.origin)
        if candidate.suffix == ".py" and is_project_local(candidate, project_root):
            return candidate

    return None


class SimCacheBase:
    def save(self, cache_key: str, result: Any, metadata: dict) -> None:
        raise NotImplementedError

    def load(self, cache_key: str) -> Any:
        raise NotImplementedError

    def exists(self, cache_key: str) -> bool:
        raise NotImplementedError

    def index(self) -> list[dict]:
        raise NotImplementedError

    def add_index_entry(self, metadata: dict) -> None:
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError

    def delete(self, cache_key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class SimFileCache(SimCacheBase):
    """File-system cache: ``cache_dir/results/{key}.pkl`` + ``cache_dir/index.cache``.

    Index is held in memory after first read. Writes use atomic rename
    to prevent corruption on crash.
    """

    def __init__(self, directory, save=None, load=None):
        self._dir = Path(directory)
        self._results_dir = self._dir / "results"
        self._index_path = self._dir / "index.cache"
        if save is not None and load is not None:
            self._save, self._load = save, load
        elif save is None and load is None:
            self._save, self._load = _make_default_serializers()
        else:
            raise ValueError("save and load must both be provided or both omitted")
        self._index_cache: list[dict] | None = None

    def _ensure_dirs(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(exist_ok=True)

    def _result_path(self, cache_key: str) -> Path:
        return self._results_dir / f"{cache_key}.pkl"

    def _read_index(self) -> list[dict]:
        if self._index_cache is not None:
            return self._index_cache
        if not self._index_path.exists():
            self._index_cache = []
            return self._index_cache
        self._index_cache = self._load(str(self._index_path))
        return self._index_cache

    def _write_index(self, entries: list[dict]) -> None:
        tmp = self._index_path.with_suffix(".cache.tmp")
        self._save(str(tmp), entries)
        os.replace(str(tmp), str(self._index_path))
        self._index_cache = entries

    def exists(self, cache_key: str) -> bool:
        return self._result_path(cache_key).exists()

    def save(self, cache_key: str, result: Any, metadata: dict) -> None:
        # content-addressed: existing file is canonical by definition, skip overwrite
        self._ensure_dirs()
        path = self._result_path(cache_key)
        if not path.exists():
            self._save(str(path), result)
        self.add_index_entry({**metadata, "cache_key": cache_key})

    def load(self, cache_key: str) -> Any:
        path = self._result_path(cache_key)
        if not path.exists():
            raise KeyError(f"no cached result for key {cache_key!r}")
        return self._load(str(path))

    def add_index_entry(self, metadata: dict) -> None:
        # supports cross-set deduplication: register an index entry for a
        # result already cached by another parameter set without duplicating
        # the file
        entries = self._read_index()
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata,
        }
        entries.append(entry)
        self._write_index(entries)

    def index(self) -> list[dict]:
        return list(self._read_index())

    def keys(self) -> list[str]:
        # derived from filenames, not the index, so it reflects actual disk
        # state even if the index is stale
        if not self._results_dir.exists():
            return []
        return [p.stem for p in self._results_dir.glob("*.pkl")]

    def delete(self, cache_key: str) -> None:
        path = self._result_path(cache_key)
        if path.exists():
            path.unlink()

        entries = self._read_index()
        filtered = [e for e in entries if e.get("cache_key") != cache_key]
        if len(filtered) != len(entries):
            self._write_index(filtered)

    def clear(self) -> None:
        if self._results_dir.exists():
            for p in self._results_dir.glob("*.pkl"):
                p.unlink()
        if self._index_path.exists():
            self._index_path.unlink()
        self._index_cache = None

    def get_fn_hash(self, cache_key: str) -> "str | None":
        # scan in reverse to return the most recent entry for this key
        for entry in reversed(self._read_index()):
            if entry.get("cache_key") == cache_key:
                return entry.get("fn_hash")
        return None
