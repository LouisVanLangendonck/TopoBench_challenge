#!/usr/bin/env python3
"""Train-split topology counts for graph datasets (with lifting) and native simplicial datasets.

Graph datasets (``graph/...``): same train split as training, **lifting only** (no HOPSE
encodings) using ``CellCycleLifting`` or ``SimplicialCliqueLifting`` with neighborhoods
from ``configs/model/{cell,simplicial}/hopse_m.yaml``.

Native simplicial datasets (``simplicial/...``, e.g. MANTRA): **no lifting** --- data are
already simplicial complexes. We report **edges** (from ``edge_index``, undirected when
stored bidirectionally) and **triangles** (count of 2-simplices from ``shape[2]``, same
as manifold 2-skeleton). Cell lifting is not applied.

Default dataset list matches ``DATASETS`` in ``scripts/hopse_plotting/main_loader.py``.

Training is started like ``python -m topobench dataset=graph/MUTAG ...`` (see e.g.
``scripts/best_val_reruns_parallel.sh``). Dataset loaders use YAML ``_target_`` strings
such as ``topobench.data.loaders.TUDatasetLoader``; Hydra's default locator can fail when
an installed ``topobench`` wheel shadows this checkout, because those classes live in
defining submodules. This script therefore instantiates loaders via those modules directly.

Writes:
  * a summary CSV (one row per graph dataset x lifting domain, or one row per native simplicial dataset)
  * a per-graph CSV (see ``--per-graph-csv`` / ``--no-per-graph``)
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import statistics
import sys
from pathlib import Path

import rootutils
from omegaconf import OmegaConf
from torch_geometric.utils import is_undirected

# Project root (.../07_topobench_fork)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Hydra must resolve ``topobench.data.loaders.*`` from this repo (not only an installed wheel).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match training: PROJECT_ROOT for dataset paths in YAML
os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))

# Register minimal OmegaConf resolvers used by dataset YAMLs
OmegaConf.register_new_resolver(
    "oc.env", lambda key: os.environ.get(str(key), ""), replace=True
)
# Do NOT register ``OmegaConf.select`` as ``oc.select`` (wrong API; breaks Mantra YAML).
if not OmegaConf.has_resolver("oc.select"):
    try:
        from omegaconf.resolvers import oc as _omega_conf_oc

        OmegaConf.register_new_resolver("oc.select", _omega_conf_oc.select, replace=True)
    except (ImportError, AttributeError):
        pass


def _load_config_resolvers():
    """Load ``set_preserve_edge_attr`` without importing ``topobench`` package ``__init__``."""
    import importlib.util

    path = REPO_ROOT / "topobench/utils/config_resolvers.py"
    spec = importlib.util.spec_from_file_location("_tb_config_resolvers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_cr = _load_config_resolvers()
OmegaConf.register_new_resolver(
    "set_preserve_edge_attr",
    _cr.set_preserve_edge_attr,
    replace=True,
)

# Keep in sync with scripts/hopse_plotting/main_loader.py DATASETS
DEFAULT_DATASETS = [
    "graph/MUTAG",
    "graph/PROTEINS",
    "graph/NCI1",
    "graph/NCI109",
    "simplicial/mantra_name",
    "simplicial/mantra_orientation",
    "simplicial/mantra_betti_numbers",
    "graph/BBB_Martins",
    "graph/CYP3A4_Veith",
    "graph/Clearance_Hepatocyte_AZ",
    "graph/Caco2_Wang",
]


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """Return (domain, config_stem), e.g. (\"graph\", \"MUTAG\") or (\"simplicial\", \"mantra_name\")."""
    s = spec.strip()
    if "/" in s:
        dom, stem = s.split("/", 1)
        return dom, stem
    return "graph", s


# YAML uses aggregate ``topobench.data.loaders.<Name>``; map to defining module (Hydra-safe).
_LOADER_TARGET_ALIASES: dict[str, tuple[str, str]] = {
    "topobench.data.loaders.TUDatasetLoader": (
        "topobench.data.loaders.graph.tu_datasets",
        "TUDatasetLoader",
    ),
    "topobench.data.loaders.ADMEDatasetLoader": (
        "topobench.data.loaders.graph.adme_datasets",
        "ADMEDatasetLoader",
    ),
    "topobench.data.loaders.MantraSimplicialDatasetLoader": (
        "topobench.data.loaders.simplicial.mantra_dataset_loader",
        "MantraSimplicialDatasetLoader",
    ),
    "topobench.data.loaders.PlanetoidDatasetLoader": (
        "topobench.data.loaders.graph.planetoid_datasets",
        "PlanetoidDatasetLoader",
    ),
    "topobench.data.loaders.ManualGraphDatasetLoader": (
        "topobench.data.loaders.graph.manual_graph_dataset_loader",
        "ManualGraphDatasetLoader",
    ),
    "topobench.data.loaders.HeterophilousGraphDatasetLoader": (
        "topobench.data.loaders.graph.hetero_datasets",
        "HeterophilousGraphDatasetLoader",
    ),
    "topobench.data.loaders.MoleculeDatasetLoader": (
        "topobench.data.loaders.graph.molecule_datasets",
        "MoleculeDatasetLoader",
    ),
    "topobench.data.loaders.graph.MoleculeDatasetLoader": (
        "topobench.data.loaders.graph.molecule_datasets",
        "MoleculeDatasetLoader",
    ),
    "topobench.data.loaders.OGBGDatasetLoader": (
        "topobench.data.loaders.graph.ogbg_datasets",
        "OGBGDatasetLoader",
    ),
    "topobench.data.loaders.USCountyDemosDatasetLoader": (
        "topobench.data.loaders.graph.us_county_demos_dataset_loader",
        "USCountyDemosDatasetLoader",
    ),
    "topobench.data.loaders.HypergraphDatasetLoader": (
        "topobench.data.loaders.hypergraph.hypergraph_dataset_loader",
        "HypergraphDatasetLoader",
    ),
    "topobench.data.loaders.CitationHypergraphDatasetLoader": (
        "topobench.data.loaders.hypergraph.citation_hypergraph_dataset_loader",
        "CitationHypergraphDatasetLoader",
    ),
    "topobench.data.loaders.GeometricShapesDatasetLoader": (
        "topobench.data.loaders.pointcloud.geometric_shapes",
        "GeometricShapesDatasetLoader",
    ),
}


def _instantiate_dataset_loader(loader_cfg) -> object:
    """Construct a dataset loader like ``hydra.utils.instantiate(cfg.dataset.loader)``.

    Avoids Hydra ``_locate`` on ``topobench.data.loaders.<Class>`` (re-exports / wheel shadowing).
    """
    target = OmegaConf.select(loader_cfg, "_target_")
    if target is None:
        raise ValueError("dataset.loader is missing _target_")
    target = str(target)
    params = loader_cfg.parameters

    if target in _LOADER_TARGET_ALIASES:
        mod_name, cls_name = _LOADER_TARGET_ALIASES[target]
    else:
        mod_name, _, cls_name = target.rpartition(".")
        if not mod_name:
            raise ValueError(f"Invalid loader _target_: {target!r}")

    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(params)


def _undirected_edge_count(data) -> int:
    """Count edges from ``edge_index`` (undirected unique count when PyG marks undirected)."""
    if not hasattr(data, "edge_index") or data.edge_index is None:
        return 0
    ei = data.edge_index
    n = int(ei.size(1))
    if n == 0:
        return 0
    try:
        if is_undirected(ei):
            return n // 2
    except Exception:
        pass
    return n


def _rank2_count(data) -> int | None:
    """Count at rank 2 from ``shape``: 2-simplices (triangles) or 2-cells after lifting."""
    if not hasattr(data, "shape") or data.shape is None:
        return None
    sh = data.shape
    if hasattr(sh, "tolist"):
        sh = sh.tolist()
    if len(sh) <= 2:
        return 0
    return int(sh[2])


def _build_paths_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "paths": {
                "root_dir": str(REPO_ROOT),
                "data_dir": str(REPO_ROOT / "datasets"),
                "log_dir": str(REPO_ROOT / "logs"),
                "output_dir": str(REPO_ROOT / "outputs" / "lift_stats"),
                "work_dir": str(Path.cwd()),
            }
        }
    )


def _load_hopse_m_yaml(model_domain: str) -> OmegaConf:
    return OmegaConf.load(REPO_ROOT / f"configs/model/{model_domain}/hopse_m.yaml")


def _load_hopse_m_neighborhoods(model_domain: str) -> list:
    m = _load_hopse_m_yaml(model_domain)
    pp = m.get("preprocessing_params")
    if pp is None:
        return []
    n = pp.get("neighborhoods")
    if n is None:
        return []
    return list(OmegaConf.to_container(n, resolve=True))


def _max_dim_from_dataset(ds: OmegaConf) -> int:
    v = OmegaConf.select(ds, "parameters.max_dim_if_lifted")
    return int(v) if v is not None else 2


def _preserve_edge_attr_for_hopse_m() -> bool:
    return bool(_cr.set_preserve_edge_attr("hopse_m", True))


def _lifting_transform_cfg(model_domain: str, ds: OmegaConf) -> OmegaConf:
    neighborhoods = _load_hopse_m_neighborhoods(model_domain)
    preserve = _preserve_edge_attr_for_hopse_m()

    if model_domain == "simplicial":
        return OmegaConf.create(
            {
                "transform_type": "lifting",
                "transform_name": "SimplicialCliqueLifting",
                "complex_dim": _max_dim_from_dataset(ds),
                "preserve_edge_attr": preserve,
                "signed": False,
                "feature_lifting": "ProjectionSum",
                "neighborhoods": neighborhoods,
            }
        )

    if model_domain == "cell":
        return OmegaConf.create(
            {
                "transform_type": "lifting",
                "transform_name": "CellCycleLifting",
                "complex_dim": 2,
                "max_cell_length": 10,
                "feature_lifting": "ProjectionSum",
                "preserve_edge_attr": preserve,
                "neighborhoods": neighborhoods,
            }
        )

    raise ValueError(f"model_domain must be 'simplicial' or 'cell', got {model_domain!r}")


def _merge_graph_cfg(graph_stem: str) -> OmegaConf:
    ds_path = REPO_ROOT / f"configs/dataset/graph/{graph_stem}.yaml"
    if not ds_path.is_file():
        raise FileNotFoundError(f"No dataset config: {ds_path}")
    ds = OmegaConf.load(ds_path)
    return OmegaConf.merge(_build_paths_cfg(), OmegaConf.create({"dataset": ds}))


def _merge_simplicial_cfg(simplicial_stem: str) -> OmegaConf:
    """Paths + dataset + ``model`` stub so Mantra loader YAML interpolations resolve."""
    ds_path = REPO_ROOT / f"configs/dataset/simplicial/{simplicial_stem}.yaml"
    if not ds_path.is_file():
        raise FileNotFoundError(f"No dataset config: {ds_path}")
    ds = OmegaConf.load(ds_path)
    model_cfg = _load_hopse_m_yaml("simplicial")
    cfg = OmegaConf.merge(
        _build_paths_cfg(),
        OmegaConf.create({"dataset": ds, "model": model_cfg}),
    )
    # Replace interpolations that depend on ``oc.select`` / optional backbone fields.
    n_list = _load_hopse_m_neighborhoods("simplicial")
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.loader.parameters.model_domain = "simplicial"
    cfg.dataset.loader.parameters.neighborhoods = OmegaConf.create(n_list)
    return cfg


def _collect_stats_graph_lifted(
    dataset_spec: str, graph_stem: str, model_domain: str
) -> tuple[dict, list[dict]]:
    from topobench.data.preprocessor import PreProcessor

    cfg = _merge_graph_cfg(graph_stem)
    if OmegaConf.select(cfg, "dataset.parameters.task_level") != "graph":
        raise ValueError("Only task_level=graph datasets are supported for lifting.")
    if OmegaConf.select(cfg, "dataset.split_params.learning_setting") != "inductive":
        raise ValueError("Only inductive splits are supported.")

    transform_cfg = _lifting_transform_cfg(model_domain, cfg.dataset)
    loader = _instantiate_dataset_loader(cfg.dataset.loader)
    dataset, dataset_dir = loader.load()

    preprocessor = PreProcessor(dataset, dataset_dir, transform_cfg)
    train_ds, _, _ = preprocessor.load_dataset_splits(cfg.dataset.split_params)

    per_graph: list[dict] = []
    edges_list: list[int] = []
    rank2_list: list[int] = []

    for i in range(len(train_ds)):
        data = train_ds[i]
        ne = _undirected_edge_count(data)
        r2 = _rank2_count(data)
        if r2 is None:
            r2 = -1
        edges_list.append(ne)
        rank2_list.append(int(r2))
        per_graph.append(
            {
                "dataset": dataset_spec,
                "lifting_domain": model_domain,
                "train_split_index": i,
                "num_edges_undirected": ne,
                "num_rank2": int(r2),
            }
        )

    def _moments(xs: list[int]) -> dict:
        if not xs:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0,
                "max": 0,
                "std": 0.0,
            }
        return {
            "mean": float(statistics.mean(xs)),
            "median": float(statistics.median(xs)),
            "min": int(min(xs)),
            "max": int(max(xs)),
            "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        }

    e_m = _moments(edges_list)
    h_m = _moments([x for x in rank2_list if x >= 0])

    higher_label = (
        "triangles_2simplices_clique"
        if model_domain == "simplicial"
        else "two_cells_cycle_basis"
    )

    summary = {
        "dataset": dataset_spec,
        "lifting_domain": model_domain,
        "higher_order_label": higher_label,
        "train_num_graphs": len(train_ds),
        "train_total_edges_undirected": sum(edges_list),
        "train_edges_per_graph_mean": e_m["mean"],
        "train_edges_per_graph_median": e_m["median"],
        "train_edges_per_graph_min": e_m["min"],
        "train_edges_per_graph_max": e_m["max"],
        "train_edges_per_graph_std": e_m["std"],
        "train_total_rank2": sum(x for x in rank2_list if x >= 0),
        "train_rank2_per_graph_mean": h_m["mean"],
        "train_rank2_per_graph_median": h_m["median"],
        "train_rank2_per_graph_min": h_m["min"],
        "train_rank2_per_graph_max": h_m["max"],
        "train_rank2_per_graph_std": h_m["std"],
        "preprocessor_time_sec": float(preprocessor.preprocessing_time),
    }
    return summary, per_graph


def _collect_stats_native_simplicial(dataset_spec: str, simplicial_stem: str) -> tuple[dict, list[dict]]:
    """Edges + 2-simplices (triangles) from stored complex; no graph lifting."""
    from topobench.data.preprocessor import PreProcessor

    cfg = _merge_simplicial_cfg(simplicial_stem)
    if OmegaConf.select(cfg, "dataset.parameters.task_level") != "graph":
        raise ValueError("Expected task_level=graph for mantra graph-level tasks.")
    if OmegaConf.select(cfg, "dataset.split_params.learning_setting") != "inductive":
        raise ValueError("Only inductive splits are supported.")

    loader = _instantiate_dataset_loader(cfg.dataset.loader)
    dataset, dataset_dir = loader.load()

    preprocessor = PreProcessor(dataset, dataset_dir, transforms_config=None)
    train_ds, _, _ = preprocessor.load_dataset_splits(cfg.dataset.split_params)

    per_graph: list[dict] = []
    edges_list: list[int] = []
    tri_list: list[int] = []

    for i in range(len(train_ds)):
        data = train_ds[i]
        ne = _undirected_edge_count(data)
        nt = _rank2_count(data)
        if nt is None:
            nt = -1
        edges_list.append(ne)
        tri_list.append(int(nt))
        per_graph.append(
            {
                "dataset": dataset_spec,
                "lifting_domain": "native_simplicial",
                "train_split_index": i,
                "num_edges_undirected": ne,
                "num_rank2": int(nt),
            }
        )

    def _moments(xs: list[int]) -> dict:
        if not xs:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0,
                "max": 0,
                "std": 0.0,
            }
        return {
            "mean": float(statistics.mean(xs)),
            "median": float(statistics.median(xs)),
            "min": int(min(xs)),
            "max": int(max(xs)),
            "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        }

    e_m = _moments(edges_list)
    t_m = _moments([x for x in tri_list if x >= 0])

    summary = {
        "dataset": dataset_spec,
        "lifting_domain": "native_simplicial",
        "higher_order_label": "triangles_2simplices_native_no_lift",
        "train_num_graphs": len(train_ds),
        "train_total_edges_undirected": sum(edges_list),
        "train_edges_per_graph_mean": e_m["mean"],
        "train_edges_per_graph_median": e_m["median"],
        "train_edges_per_graph_min": e_m["min"],
        "train_edges_per_graph_max": e_m["max"],
        "train_edges_per_graph_std": e_m["std"],
        "train_total_rank2": sum(x for x in tri_list if x >= 0),
        "train_rank2_per_graph_mean": t_m["mean"],
        "train_rank2_per_graph_median": t_m["median"],
        "train_rank2_per_graph_min": t_m["min"],
        "train_rank2_per_graph_max": t_m["max"],
        "train_rank2_per_graph_std": t_m["std"],
        "preprocessor_time_sec": float(preprocessor.preprocessing_time),
    }
    return summary, per_graph


def main() -> int:
    # ``pyproject.toml`` / ``.git`` live at repo root (``.project-root`` is often missing).
    for indicator in ("pyproject.toml", ".git", ".project-root"):
        if (REPO_ROOT / indicator).exists():
            rootutils.setup_root(__file__, indicator=indicator, pythonpath=True)
            break
    else:
        sys.path.insert(0, str(REPO_ROOT))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "datasets",
        nargs="*",
        help="Dataset specs: graph/STEM or simplicial/STEM (default: same list as hopse_plotting/main_loader.py).",
    )
    p.add_argument(
        "--model-domains",
        nargs="+",
        default=["simplicial", "cell"],
        choices=["simplicial", "cell"],
        help="Lifting branch for graph/* datasets only (hopse_m cell vs simplicial configs).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "scripts" / "csvs" / "graph_lift_topology_summary.csv",
        help="Output path for the summary CSV.",
    )
    p.add_argument(
        "--per-graph-csv",
        type=Path,
        default=REPO_ROOT / "scripts" / "csvs" / "graph_lift_topology_per_graph.csv",
        help="Per-train-graph CSV path (long format).",
    )
    p.add_argument(
        "--no-per-graph",
        action="store_true",
        help="Do not write the per-graph CSV.",
    )
    args = p.parse_args()
    per_graph_path: Path | None = None if args.no_per_graph else args.per_graph_csv

    names = args.datasets if args.datasets else DEFAULT_DATASETS

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    per_rows: list[dict] = []

    for spec in names:
        domain, stem = parse_dataset_spec(spec)
        if domain == "graph":
            for lift_dom in args.model_domains:
                try:
                    s, pg = _collect_stats_graph_lifted(spec, stem, lift_dom)
                    summaries.append(s)
                    per_rows.extend(pg)
                    print(
                        f"OK  {spec:42} {lift_dom:10}  "
                        f"train={s['train_num_graphs']:5d}  "
                        f"edges(total)={s['train_total_edges_undirected']:8d}  "
                        f"rank2(total)={s['train_total_rank2']:8d}"
                    )
                except Exception as e:
                    print(
                        f"ERR {spec:42} {lift_dom:10}  {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
        elif domain == "simplicial":
            try:
                s, pg = _collect_stats_native_simplicial(spec, stem)
                summaries.append(s)
                per_rows.extend(pg)
                print(
                    f"OK  {spec:42} {'native':10}  "
                    f"train={s['train_num_graphs']:5d}  "
                    f"edges(total)={s['train_total_edges_undirected']:8d}  "
                    f"triangles(rank2)={s['train_total_rank2']:8d}"
                )
            except Exception as e:
                print(
                    f"ERR {spec:42} {'native':10}  {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
        else:
            print(
                f"ERR {spec:42} {'':10}  Unsupported domain {domain!r} (use graph/ or simplicial/)",
                file=sys.stderr,
            )

    if summaries:
        fieldnames = list(summaries[0].keys())
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summaries)
        print(f"Wrote summary: {args.summary_csv}")

    if per_graph_path and per_rows:
        per_graph_path.parent.mkdir(parents=True, exist_ok=True)
        pg_fields = list(per_rows[0].keys())
        with per_graph_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pg_fields)
            w.writeheader()
            w.writerows(per_rows)
        print(f"Wrote per-graph: {per_graph_path}")

    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
