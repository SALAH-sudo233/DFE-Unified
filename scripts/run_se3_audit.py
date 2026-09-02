#!/usr/bin/env python3
"""Run analytical and model-stage SE(3) audits for the DF 500K anchor."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import write_new_manifest  # noqa: E402
from dfe.diagnostics.io import sha256_file  # noqa: E402
from dfe.diagnostics.model_audit import (  # noqa: E402
    audit_analytical_df_state,
    compare_event_sets,
)
from dfe.diagnostics.observer import TensorObserver  # noqa: E402
from dfe.diagnostics.openness import parse_pdb_heavy_atoms  # noqa: E402
from dfe.diagnostics.se3 import sample_so3  # noqa: E402
from models.df_module import AnalyticalDirectionField  # noqa: E402
from dfe.science.model_precision import (  # noqa: E402
    normalize_model_dtype,
    torch_model_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--rotations", type=int, default=100)
    parser.add_argument("--translations", type=int, default=10)
    parser.add_argument("--stage", choices=("preflight", "full"), default="full")
    parser.add_argument(
        "--vector-origin-mode",
        choices=("absolute", "centered", "zero"),
        default="absolute",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line]


def _pocket_path(record: dict[str, object]) -> Path:
    item = next(value for value in record["inputs"] if value["role"] == "pocket")
    path = Path(item["source_path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"pocket input hash mismatch: {record['pocket_id']}")
    return path


def _translations(seed: int, count: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.uniform(-10.0, 10.0, size=(count, 3))


def _preflight_transforms(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[dict[str, object], ...]:
    identity = np.eye(3, dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    return (
        {"category": "identity", "rotation": identity, "translation": zero},
        {"category": "rotation", "rotation": rotation, "translation": zero},
        {
            "category": "translation",
            "rotation": identity,
            "translation": translation,
        },
        {"category": "rigid", "rotation": rotation, "translation": translation},
    )


def _full_transforms(
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "category": "rigid",
            "rotation": rotation,
            "translation": translations[index % len(translations)],
        }
        for index, rotation in enumerate(rotations)
    )


def _preflight_gate(
    records: list[dict[str, object]],
    *,
    tolerance: float,
) -> tuple[bool, str | None]:
    required_categories = ("identity", "rotation", "translation", "rigid")
    by_category = {
        str(record.get("transform_category")): record for record in records
    }
    for category in required_categories:
        if category not in by_category:
            return False, f"{category}:missing_transform"
        record = by_category[category]
        if record.get("topology_match") is not True:
            return False, f"{category}:topology.edge_index"
        events = {
            str(event.get("key")): event for event in record.get("events", [])
        }
        for event_name in ("encoder.scalar", "encoder.vector"):
            key = next(
                (item for item in events if item.endswith(f":{event_name}")),
                None,
            )
            if key is None:
                return False, f"{category}:missing:{event_name}"
            error = events[key].get("normalized_max")
            if error is None or not np.isfinite(float(error)):
                return False, f"{category}:{key}"
            if float(error) >= tolerance:
                return False, f"{category}:{key}"
    return True, None


def _current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _analytical_audit(
    pockets: list[dict[str, object]],
    transforms: tuple[dict[str, object], ...],
    tolerance: float,
) -> dict[str, object]:
    module = AnalyticalDirectionField(hidden_dim=64).double().eval()
    reports = []
    for pocket in pockets:
        atoms = parse_pdb_heavy_atoms(_pocket_path(pocket))
        center = pocket["center"]
        query = torch.tensor(
            [[center["x"], center["y"], center["z"]]], dtype=torch.float64
        )
        pocket_pos = torch.from_numpy(atoms.coordinates).double()
        pocket_types = torch.zeros(len(pocket_pos), dtype=torch.long)
        pocket_mask = torch.ones(len(pocket_pos), dtype=torch.bool)
        for transform_index, transform in enumerate(transforms):
            rotation = np.asarray(transform["rotation"])
            translation = np.asarray(transform["translation"])
            report = audit_analytical_df_state(
                module,
                query,
                pocket_pos,
                pocket_types,
                pocket_mask,
                rotation,
                translation,
                tolerance=tolerance,
            )
            reports.append(
                {
                    "pocket_id": pocket["pocket_id"],
                    "transform_index": transform_index,
                    "transform_category": transform["category"],
                    **report.to_dict(),
                }
            )
    first_failure = next(
        (record["first_failure"] for record in reports if not record["passed"]), None
    )
    return {
        "passed": first_failure is None,
        "first_failure": first_failure,
        "state_count": len(pockets),
        "comparison_count": len(reports),
        "reports": reports,
    }


def _load_model_runtime(
    checkpoint_path: Path,
    device: str,
    model_dtype: str | None = None,
):
    try:
        from torch_geometric.data import Batch
        from torch_geometric.transforms import Compose

        from models.maskfill import MaskFillModelVN
        from utils.data import FOLLOW_BATCH, ProteinLigandData, torchify_dict
        from utils.protein_ligand import PDBProtein
        from utils.transforms import (
            AtomComposer,
            FeaturizeLigandAtom,
            FeaturizeProteinAtom,
            LigandCountNeighbors,
            LigandMaskAll,
            RefineData,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Pocket2Mol model dependencies are unavailable; use the env_cuda113 environment"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    model = MaskFillModelVN(
        checkpoint["config"].model,
        num_classes=7,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        num_bond_types=3,
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"checkpoint state mismatch: {incompatible}")
    selected_dtype = normalize_model_dtype(model_dtype)
    model.to(device=device, dtype=torch_model_dtype(selected_dtype))
    model.eval()
    transform = Compose(
        [
            RefineData(),
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            LigandMaskAll(),
        ]
    )
    composer = AtomComposer(
        protein_featurizer.feature_dim,
        ligand_featurizer.feature_dim,
        model.config.encoder.knn,
    )
    return {
        "model": model,
        "transform": transform,
        "composer": composer,
        "Batch": Batch,
        "FOLLOW_BATCH": FOLLOW_BATCH,
        "ProteinLigandData": ProteinLigandData,
        "PDBProtein": PDBProtein,
        "torchify_dict": torchify_dict,
        "model_dtype": selected_dtype,
        "checkpoint_strict_load": True,
    }


def _base_data(runtime, pocket_path: Path):
    protein_dict = runtime["torchify_dict"](
        runtime["PDBProtein"](str(pocket_path)).to_dict_atom()
    )
    data = runtime["ProteinLigandData"].from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict={
            "element": torch.empty((0,), dtype=torch.long),
            "pos": torch.empty((0, 3), dtype=torch.float32),
            "atom_feature": torch.empty((0, 8), dtype=torch.float32),
            "bond_index": torch.empty((2, 0), dtype=torch.long),
            "bond_type": torch.empty((0,), dtype=torch.long),
        },
    )
    return runtime["transform"](data)


def _run_model_state(runtime, base_data, positions: torch.Tensor, device: str):
    data = copy.deepcopy(base_data)
    model_dtype = torch_model_dtype(runtime["model_dtype"])
    data.protein_pos = positions.to(dtype=model_dtype).cpu()
    data = runtime["composer"](data)
    batch = runtime["Batch"].from_data_list(
        [data], follow_batch=runtime["FOLLOW_BATCH"]
    ).to(device)
    observer = TensorObserver()
    model = runtime["model"]
    model.set_diagnostics(observer=observer.at_step(0))
    with torch.no_grad():
        model.sample_init(
            compose_feature=batch.compose_feature.to(dtype=model_dtype),
            compose_pos=batch.compose_pos.to(dtype=model_dtype),
            idx_protein=batch.idx_protein_in_compose,
            compose_knn_edge_index=batch.compose_knn_edge_index,
            compose_knn_edge_feature=batch.compose_knn_edge_feature.to(
                dtype=model_dtype
            ),
            n_samples_pos=-1,
            n_samples_atom=-1,
        )
    model.set_diagnostics()
    return observer, batch.compose_knn_edge_index.detach().cpu().clone()


def _model_audit(
    pockets: list[dict[str, object]],
    transforms: tuple[dict[str, object], ...],
    checkpoint_path: Path,
    device: str,
    tolerance: float,
    stage: str,
    vector_origin_mode: str,
    model_dtype: str,
) -> dict[str, object]:
    runtime = _load_model_runtime(checkpoint_path, device, model_dtype)
    reports = []
    model = runtime["model"]
    model.set_science_vector_origin(vector_origin_mode)
    try:
        for pocket in pockets:
            base_data = _base_data(runtime, _pocket_path(pocket))
            positions = base_data.protein_pos.to(
                dtype=torch_model_dtype(runtime["model_dtype"])
            )
            reference, reference_edges = _run_model_state(
                runtime,
                base_data,
                positions,
                device,
            )
            for transform_index, transform in enumerate(transforms):
                rotation = np.asarray(transform["rotation"])
                translation = np.asarray(transform["translation"])
                rotation_tensor = torch.as_tensor(
                    rotation,
                    dtype=positions.dtype,
                )
                translation_tensor = torch.as_tensor(
                    translation,
                    dtype=positions.dtype,
                )
                moved = positions @ rotation_tensor.T + translation_tensor
                transformed, transformed_edges = _run_model_state(
                    runtime,
                    base_data,
                    moved,
                    device,
                )
                topology_match = torch.equal(
                    reference_edges,
                    transformed_edges,
                )
                report = compare_event_sets(
                    reference,
                    transformed,
                    rotation,
                    translation,
                    tolerance=tolerance,
                ).to_dict()
                if not topology_match:
                    report["passed"] = False
                    report["first_failure"] = "topology.edge_index"
                reports.append(
                    {
                        "pocket_id": pocket["pocket_id"],
                        "transform_index": transform_index,
                        "transform_category": transform["category"],
                        "topology_match": topology_match,
                        **report,
                    }
                )
    finally:
        model.set_science_vector_origin()
    if stage == "preflight":
        passed, first_failure = _preflight_gate(
            reports,
            tolerance=tolerance,
        )
    else:
        first_failure = next(
            (
                f"{record['pocket_id']}:{record['first_failure']}"
                for record in reports
                if not record["passed"]
            ),
            None,
        )
        passed = first_failure is None
    return {
        "passed": passed,
        "first_failure": first_failure,
        "state_count": len(pockets),
        "comparison_count": len(reports),
        "checkpoint_strict_load": runtime["checkpoint_strict_load"],
        "vector_origin_mode": vector_origin_mode,
        "model_dtype": runtime["model_dtype"],
        "reports": reports,
    }


def run(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    run_root = manifest_path.parent
    pockets_path = run_root / manifest["artifacts"]["pockets"]["path"]
    if sha256_file(pockets_path) != manifest["artifacts"]["pockets"]["sha256"]:
        raise ValueError("pockets.jsonl hash does not match the run manifest")
    checkpoint = manifest["inputs"]["checkpoint"]
    checkpoint_path = Path(checkpoint["source_path"])
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("checkpoint hash does not match the run manifest")
    if args.rotations != manifest["protocol"]["se3"]["rotations"]:
        raise ValueError("rotation count differs from the frozen manifest")
    if args.translations != manifest["protocol"]["se3"]["translations"]:
        raise ValueError("translation count differs from the frozen manifest")
    all_pockets = _read_jsonl(pockets_path)[:20]
    if len(all_pockets) < 20:
        raise ValueError("model SE(3) audit requires at least 20 manifest pockets")
    rotations = sample_so3(20260901, args.rotations)
    translations = _translations(20260901, args.translations)
    stage = getattr(args, "stage", "full")
    vector_origin_mode = getattr(args, "vector_origin_mode", "absolute")
    model_dtype = normalize_model_dtype(getattr(args, "model_dtype", "float32"))
    if stage == "preflight":
        pockets = all_pockets[:1]
        transforms = _preflight_transforms(rotations[0], translations[0])
    elif stage == "full":
        pockets = all_pockets
        transforms = _full_transforms(rotations, translations)
    else:
        raise ValueError(f"unknown SCI-1 stage: {stage}")
    se3_protocol = manifest["protocol"]["se3"]
    analytical = _analytical_audit(
        pockets,
        transforms,
        float(se3_protocol["analytical_float64_tolerance"]),
    )
    model = _model_audit(
        pockets,
        transforms,
        checkpoint_path,
        args.device,
        float(se3_protocol["model_float32_tolerance"]),
        stage,
        vector_origin_mode,
        model_dtype,
    )
    passed = bool(analytical["passed"] and model["passed"])
    return (
        0 if passed else 2,
        {
            "schema_version": manifest["schema_version"],
            "manifest_hash": manifest["manifest_hash"],
            "source_commit": _current_commit(),
            "input_source_commit": manifest.get("git", {}).get("commit"),
            "status": "pass" if passed else "scientific_failure",
            "device": args.device,
            "stage": stage,
            "vector_origin_mode": vector_origin_mode,
            "model_dtype": model_dtype,
            "checkpoint_sha256": checkpoint["sha256"],
            "rotations": args.rotations,
            "translations": args.translations,
            "analytical": analytical,
            "model": model,
        },
    )


def main() -> int:
    args = parse_args()
    output = (args.output or args.manifest.resolve().parent / "se3-audit.json").resolve()
    if output.exists():
        raise FileExistsError(output)
    try:
        exit_code, report = run(args)
    except Exception as exc:
        report = {
            "status": "infrastructure_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_new_manifest(output, report)
        print(f"SE(3) audit infrastructure failure: {exc}", file=sys.stderr)
        return 1
    write_new_manifest(output, report)
    print(f"SE(3) audit status: {report['status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
