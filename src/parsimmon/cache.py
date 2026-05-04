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
            r2 = repr(copy.deepcopy(obj))
        except Exception:
            raise TypeError(
                f"Cannot hash object of type {type(obj).__name__} for caching: "
                f"deepcopy failed, so repr stability cannot be verified."
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
    if rel.parts and rel.parts[0] in venv_markers:
        return False

    return True


def find_project_root(start: Path) -> Path:
    # .git as root marker so the hash boundary matches version control;
    # falls back to parent of start for notebooks / test scripts
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve().parent


def hash_function_chain(fn: Callable) -> str:
    """AST-based identity hash of fn and its project-local transitive imports.

    Uses AST dumps so whitespace/comment changes don't invalidate the
    cache, but logic changes do.
    """
    module = inspect.getmodule(fn)
    module_file = getattr(module, "__file__", None) if module else None
    if module_file is None:
        src = inspect.getsource(fn)
        return hashlib.sha256(src.encode()).hexdigest()[:16]

    project_root = find_project_root(Path(module_file))
    visited: dict[str, str] = {}  # module_path -> ast_dump
    _collect_module_asts(Path(module_file), project_root, visited, seen=set())

    combined = "\n".join(visited[k] for k in sorted(visited)).encode()
    return hashlib.sha256(combined).hexdigest()[:16]


def _collect_module_asts(module_path: Path, project_root: Path, visited: dict, seen: set) -> None:
    abs_path = str(module_path.resolve())
    if abs_path in seen:
        return
    seen.add(abs_path)

    if not is_project_local(module_path, project_root):
        return

    try:
        source = module_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    visited[abs_path] = ast.dump(tree)

    for node in ast.walk(tree):
        dep_path = _resolve_import_node(node, project_root)
        if dep_path is not None:
            _collect_module_asts(dep_path, project_root, visited, seen)


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
        self._index_cache: "list[dict] | None" = None

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
        self._ensure_dirs()
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
        for entry in self._read_index():
            if entry.get("cache_key") == cache_key:
                return entry.get("fn_hash")
        return None
