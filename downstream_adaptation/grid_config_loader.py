"""YAML configs for inductive / transductive downstream grid scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

GRID_CONFIG_DIR = Path(__file__).resolve().parent / "grid_configs"


def load_grid_yaml(path: str | Path) -> dict[str, Any]:
    """Load a grid YAML file. Returns {} if file is empty."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Grid config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def coalesce(*values):
    """Return the first value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


def build_worker_devices(
    parallel_workers: int,
    fallback_device: str,
    eval_devices: list[str] | None,
) -> list[str]:
    """
    Device string for each worker slot (length == ``parallel_workers``).

    When ``parallel_workers == 1``, returns ``[fallback_device]``.
    When ``parallel_workers > 1`` and ``eval_devices`` is None, uses
    ``cuda:0`` .. ``cuda:{parallel_workers - 1}`` if ``fallback_device`` looks like CUDA.
    """
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be >= 1")
    if parallel_workers == 1:
        return [fallback_device]
    if eval_devices is not None:
        if len(eval_devices) != parallel_workers:
            raise ValueError(
                "eval_devices must have the same length as parallel_workers "
                f"(got {len(eval_devices)} devices, parallel_workers={parallel_workers})"
            )
        return list(eval_devices)
    fb = fallback_device.strip().lower()
    if fb.startswith("cpu"):
        raise ValueError(
            "parallel_workers > 1 is not supported with CPU; set parallel_workers: 1 "
            "or pass explicit CUDA eval_devices."
        )
    if fb.startswith("cuda"):
        return [f"cuda:{i}" for i in range(parallel_workers)]
    raise ValueError(
        f"Cannot infer eval_devices from device={fallback_device!r}; "
        "set eval_devices in YAML or use parallel_workers: 1."
    )


def coerce_optional_int_list(field: str, val: Any) -> list[int] | None:
    """Accept YAML ``null``, a single int, or a list of ints (e.g. ``n_train`` / inductive sizes)."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError(f"{field}: bool is not allowed; use integers or null")
    if isinstance(val, int):
        return [val]
    if isinstance(val, list):
        out: list[int] = []
        for i, x in enumerate(val):
            if not isinstance(x, int) or isinstance(x, bool):
                raise TypeError(f"{field}[{i}] must be int, got {x!r}")
            out.append(x)
        return out
    raise TypeError(f"{field} must be null, int, or list[int]; got {type(val).__name__}")


def coerce_optional_float_list(field: str, val: Any) -> list[float] | None:
    """Accept YAML ``null``, a single float/int, or a list of floats/ints (e.g. ``lr``, ``classifier_dropout``)."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError(f"{field}: bool is not allowed; use floats or null")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return [float(val)]
    if isinstance(val, list):
        out: list[float] = []
        for i, x in enumerate(val):
            if isinstance(x, bool):
                raise TypeError(f"{field}[{i}] must be float/int, got bool")
            if not isinstance(x, (int, float)):
                raise TypeError(f"{field}[{i}] must be float/int, got {x!r}")
            out.append(float(x))
        return out
    raise TypeError(f"{field} must be null, float/int, or list[float/int]; got {type(val).__name__}")


def coerce_optional_str_list(field: str, val: Any) -> list[str] | None:
    """Like ``coerce_str_list`` but allow YAML ``null`` (returns None)."""
    if val is None:
        return None
    return coerce_str_list(field, val)


def coerce_str_list(field: str, val: Any) -> list[str]:
    """Accept a YAML scalar or list (e.g. ``tasks: foo`` vs ``tasks: [foo, bar]``)."""
    if val is None:
        raise ValueError(f"{field} is null or missing")
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        if not val:
            raise ValueError(f"{field}: empty list")
        return [str(x) for x in val]
    raise TypeError(f"{field} must be str or list[str]; got {type(val).__name__}")


def merge_effective_config(
    file_cfg: dict[str, Any],
    *,
    defaults: dict[str, Any],
    cli: dict[str, Any],
    keys: list[str],
) -> dict[str, Any]:
    """
    Build effective config: CLI (non-None) overrides file_cfg overrides defaults.

    `cli` should map key -> value from argparse (use None when argument omitted).
    """
    out: dict[str, Any] = {}
    for key in keys:
        out[key] = coalesce(cli.get(key), file_cfg.get(key), defaults.get(key))
    return out


def normalize_graphuniverse_overrides(
    raw: Any,
    *,
    default_preset: list | None = None,
) -> list[dict | None]:
    """
    Normalize overrides from YAML or Python.

    - None / missing -> default_preset (or [None] if default_preset is None)
    - list of dicts and/or null -> list[dict | None]
    """
    if raw is None:
        if default_preset is not None:
            return deepcopy(default_preset)
        return [None]

    if not isinstance(raw, list):
        raise TypeError("graphuniverse_overrides must be a list of dicts or null entries")

    out: list[dict | None] = []
    for item in raw:
        if item is None or item == "null":
            out.append(None)
        elif isinstance(item, dict):
            out.append(deepcopy(item))
        else:
            raise TypeError(
                f"Each graphuniverse_overrides entry must be a dict or null, got {type(item)}"
            )
    return out
