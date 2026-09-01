#!/usr/bin/env python3
"""Run one manifest-declared DF 500K diagnostic generation job."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfe.diagnostics.contracts import AttemptRecord, canonical_json  # noqa: E402
from dfe.diagnostics.interventions import DFIntervention  # noqa: E402
from dfe.diagnostics.io import sha256_file  # noqa: E402
from dfe.diagnostics.ledger import AttemptLedger, replay_ledger  # noqa: E402
from dfe.diagnostics.observer import TensorObserver  # noqa: E402
from dfe.diagnostics.trace import TraceEvent, TraceWriter, summarize_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line]


def load_declared_job(
    run_root: Path,
    manifest: Mapping[str, object],
    job_id: str,
) -> tuple[dict[str, object], Path]:
    jobs_path = Path(run_root) / manifest["artifacts"]["jobs"]["path"]
    jobs = {str(record["job_id"]): record for record in _read_jsonl(jobs_path)}
    if job_id not in jobs:
        raise ValueError(f"job is not declared by the manifest: {job_id}")
    output = Path(run_root) / "jobs" / job_id
    return jobs[job_id], output


def _attempt_payload(
    run_id: str,
    job: Mapping[str, object],
    sample_index: int,
    status: str,
    **fields: object,
) -> dict[str, object]:
    record = AttemptRecord.new(
        run_id,
        str(job["pocket_id"]),
        int(job["seed"]),
        str(job["arm_id"]),
        sample_index,
    ).to_dict()
    record.update(
        {
            "schema_version": "phase0.v1",
            "job_id": job["job_id"],
            "intervention": job["intervention"],
            "arm_id": job["arm_id"],
            "status": status,
            "rng_seed": int(job["seed"]) + sample_index,
            **fields,
        }
    )
    return record


def predeclare_attempts(
    path: Path,
    run_id: str,
    job: Mapping[str, object],
) -> None:
    with AttemptLedger(path) as ledger:
        for sample_index in range(int(job["attempt_count"])):
            ledger.append(
                _attempt_payload(run_id, job, sample_index, "requested")
            )


def close_requested_after_bootstrap_failure(
    path: Path,
    run_id: str,
    job: Mapping[str, object],
    message: str,
) -> None:
    with AttemptLedger(path, resume=True) as ledger:
        for sample_index in range(int(job["attempt_count"])):
            attempt_id = AttemptRecord.new(
                run_id,
                str(job["pocket_id"]),
                int(job["seed"]),
                str(job["arm_id"]),
                sample_index,
            ).attempt_id
            if ledger.states.get(attempt_id) == "requested":
                _append_failure(
                    ledger,
                    run_id,
                    job,
                    sample_index,
                    "runtime_error",
                    message,
                )


def classify_initial_queue(queue, finished_status: str) -> str | None:
    if not queue:
        return "init_threshold_exhausted"
    if all(
        candidate.status == finished_status
        and len(getattr(candidate, "ligand_context_pos", ())) == 0
        for candidate in queue
    ):
        return "init_no_frontier"
    return None


def parity_projection(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": record.get("status"),
        "error_code": record.get("error_code"),
        "has_smiles": bool(record.get("smiles")),
    }


def _write_sdf(chem, molecule, path: Path) -> None:
    chem.MolToMolFile(molecule, str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError("SDF writer did not create a non-empty file")


def _atomic_checkpoint(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_runtime(manifest: Mapping[str, object], device: str):
    try:
        from easydict import EasyDict
        from rdkit import Chem
        from torch_geometric.transforms import Compose

        from models.maskfill import MaskFillModelVN
        from sample import (
            STATUS_FINISHED,
            STATUS_RUNNING,
            get_init,
            get_next,
            logp_to_rank_prob,
        )
        from utils.data import ProteinLigandData, torchify_dict
        from utils.datasets import transform_data
        from utils.misc import seed_all
        from utils.protein_ligand import PDBProtein
        from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges
        from utils.transforms import (
            AtomComposer,
            FeaturizeLigandAtom,
            FeaturizeProteinAtom,
            LigandCountNeighbors,
            LigandMaskAll,
            RefineData,
        )
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Pocket2Mol runtime dependencies are unavailable; use env_cuda113"
        ) from exc

    policy_record = manifest["inputs"]["sampling_policy"]
    policy_path = Path(policy_record["source_path"])
    if sha256_file(policy_path) != policy_record["sha256"]:
        raise ValueError("sampling policy hash does not match the run manifest")
    policy = EasyDict(yaml.safe_load(policy_path.read_text(encoding="utf-8")))
    checkpoint_record = manifest["inputs"]["checkpoint"]
    checkpoint_path = Path(checkpoint_record["source_path"])
    if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
        raise ValueError("checkpoint hash does not match the run manifest")
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
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    initial_transform = Compose(
        [
            RefineData(),
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
        ]
    )
    composer = AtomComposer(
        protein_featurizer.feature_dim,
        ligand_featurizer.feature_dim,
        model.config.encoder.knn,
    )
    masking = Compose([LigandMaskAll(), composer])
    return {
        "Chem": Chem,
        "MolReconsError": MolReconsError,
        "PDBProtein": PDBProtein,
        "ProteinLigandData": ProteinLigandData,
        "STATUS_FINISHED": STATUS_FINISHED,
        "STATUS_RUNNING": STATUS_RUNNING,
        "composer": composer,
        "get_init": get_init,
        "get_next": get_next,
        "initial_transform": initial_transform,
        "ligand_featurizer": ligand_featurizer,
        "logp_to_rank_prob": logp_to_rank_prob,
        "masking": masking,
        "model": model,
        "policy": policy,
        "protein_featurizer": protein_featurizer,
        "reconstruct": reconstruct_from_generated_with_edges,
        "seed_all": seed_all,
        "torchify_dict": torchify_dict,
        "transform_data": transform_data,
    }


def _empty_ligand() -> dict[str, torch.Tensor]:
    return {
        "element": torch.empty((0,), dtype=torch.long),
        "pos": torch.empty((0, 3), dtype=torch.float32),
        "atom_feature": torch.empty((0, 8), dtype=torch.float32),
        "bond_index": torch.empty((2, 0), dtype=torch.long),
        "bond_type": torch.empty((0,), dtype=torch.long),
    }


def _build_pocket_data(runtime, pocket_record: Mapping[str, object]):
    pocket_input = next(item for item in pocket_record["inputs"] if item["role"] == "pocket")
    pocket_path = Path(pocket_input["source_path"])
    if sha256_file(pocket_path) != pocket_input["sha256"]:
        raise ValueError(f"pocket input hash mismatch: {pocket_record['pocket_id']}")
    protein = runtime["torchify_dict"](
        runtime["PDBProtein"](str(pocket_path)).to_dict_atom()
    )
    data = runtime["ProteinLigandData"].from_protein_ligand_dicts(
        protein_dict=protein,
        ligand_dict=_empty_ligand(),
    )
    return runtime["initial_transform"](data)


def _pocket_by_id(run_root: Path, manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    path = run_root / manifest["artifacts"]["pockets"]["path"]
    if sha256_file(path) != manifest["artifacts"]["pockets"]["sha256"]:
        raise ValueError("pockets.jsonl hash does not match the run manifest")
    return {str(record["pocket_id"]): record for record in _read_jsonl(path)}


def _d4_pair(run_root: Path, pocket_id: str) -> str:
    openness_path = run_root / "openness.jsonl"
    if not openness_path.is_file():
        raise ValueError("D4 requires frozen openness.jsonl pairing")
    record = next(
        (item for item in _read_jsonl(openness_path) if item["pocket_id"] == pocket_id),
        None,
    )
    if record is None:
        raise ValueError(f"D4 pairing is missing for {pocket_id}")
    return str(record["wrong_pocket_id"])


class AttemptTracer:
    def __init__(self, writer: TraceWriter, identity: Mapping[str, object]) -> None:
        self.writer = writer
        self.identity = dict(identity)
        self.last_ns = writer.last_monotonic_ns

    def _time(self) -> int:
        value = max(time.monotonic_ns(), self.last_ns + 1)
        self.last_ns = value
        return value

    def tensor(self, step: int, event: str, value: torch.Tensor) -> None:
        self.writer.append(
            TraceEvent.new(
                **self.identity,
                step=step,
                event=event,
                monotonic_ns=self._time(),
                tensor=summarize_tensor(value),
            ),
            durable=False,
        )

    def decision(self, step: int, event: str, value: Mapping[str, object]) -> None:
        self.writer.append(
            TraceEvent.new(
                **self.identity,
                step=step,
                event=event,
                monotonic_ns=self._time(),
                decision=dict(value),
            )
        )


def _configure_diagnostics(
    runtime,
    data,
    job: Mapping[str, object],
    step: int,
    tracer: AttemptTracer,
    wrong_data=None,
    pocket_record: Mapping[str, object] | None = None,
    wrong_record: Mapping[str, object] | None = None,
) -> None:
    arm = str(job["intervention"])
    kwargs: dict[str, object] = {}
    if arm == "D3":
        kwargs["shuffle_seed"] = int(job["seed"]) + step
    elif arm == "D4":
        if wrong_data is None or pocket_record is None or wrong_record is None:
            raise ValueError("D4 runtime pairing is incomplete")
        query = data.compose_pos.to(next(runtime["model"].parameters()).device)
        actual_center = torch.tensor(
            list(pocket_record["center"].values()), dtype=query.dtype, device=query.device
        )
        wrong_center = torch.tensor(
            list(wrong_record["center"].values()), dtype=query.dtype, device=query.device
        )
        wrong_pos = wrong_data.protein_pos.to(query.device, query.dtype)
        wrong_pos = wrong_pos - wrong_center + actual_center
        features = wrong_data.protein_atom_feature.to(query.device)
        wrong_types = features[:, :5].argmax(dim=-1)
        wrong_mask = torch.ones(len(wrong_pos), dtype=torch.bool, device=query.device)
        with torch.no_grad():
            kwargs["alternate_raw"] = runtime["model"].df_module.raw_features(
                query, wrong_pos, wrong_types, wrong_mask
            )
    elif arm == "D5":
        kwargs["gate"] = float(job["gate"])
    intervention = DFIntervention.from_arm(arm, **kwargs)
    observer = TensorObserver()

    def callback(event: str, value: torch.Tensor) -> None:
        observer.observe(step, event, value)
        tracer.tensor(step, event, value)

    runtime["model"].set_diagnostics(intervention, callback)


def _append_failure(
    ledger: AttemptLedger,
    run_id: str,
    job: Mapping[str, object],
    sample_index: int,
    error_code: str,
    message: str,
) -> None:
    ledger.append(
        _attempt_payload(
            run_id,
            job,
            sample_index,
            "failed",
            error_code=error_code,
            error_message=message[:500],
        )
    )


def recover_interrupted_attempt(
    ledger: AttemptLedger,
    run_id: str,
    job: Mapping[str, object],
    sample_index: int,
    job_root: Path,
) -> None:
    attempt_id = AttemptRecord.new(
        run_id,
        str(job["pocket_id"]),
        int(job["seed"]),
        str(job["arm_id"]),
        sample_index,
    ).attempt_id
    state = ledger.states.get(attempt_id)
    if state in {None, "requested", "evaluated", "failed"}:
        return
    message = "process stopped before an atomic attempt terminal record"
    if state == "reconstructed":
        ledger.append(
            _attempt_payload(
                run_id,
                job,
                sample_index,
                "evaluated",
                evaluation_status="failed",
                error_code="runtime_error",
                error_message=message,
            )
        )
    else:
        _append_failure(
            ledger,
            run_id,
            job,
            sample_index,
            "runtime_error",
            message,
        )
    recovery_path = Path(job_root) / "recovery.jsonl"
    with recovery_path.open("ab") as handle:
        handle.write(
            canonical_json(
                {
                    "schema_version": "phase0.v1",
                    "attempt_id": attempt_id,
                    "recovered_from": state,
                    "terminal_state": ledger.states[attempt_id],
                }
            )
        )
        handle.flush()
        os.fsync(handle.fileno())


def close_attempt_after_runtime_error(
    ledger: AttemptLedger,
    run_id: str,
    job: Mapping[str, object],
    sample_index: int,
    error_code: str,
    message: str,
) -> None:
    attempt_id = AttemptRecord.new(
        run_id,
        str(job["pocket_id"]),
        int(job["seed"]),
        str(job["arm_id"]),
        sample_index,
    ).attempt_id
    state = ledger.states.get(attempt_id)
    if state in {"evaluated", "failed"}:
        return
    if state == "reconstructed":
        ledger.append(
            _attempt_payload(
                run_id,
                job,
                sample_index,
                "evaluated",
                evaluation_status="failed",
                error_code=error_code,
                error_message=message[:500],
            )
        )
    else:
        _append_failure(
            ledger,
            run_id,
            job,
            sample_index,
            error_code,
            message,
        )


def _run_attempt(
    runtime,
    run_root: Path,
    job_root: Path,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    sample_index: int,
    ledger: AttemptLedger,
    trace_writer: TraceWriter,
    pocket_record: Mapping[str, object],
    wrong_record: Mapping[str, object] | None,
    base_data,
    wrong_data,
    known_smiles: set[str],
    device: str,
    diagnostics_enabled: bool = True,
) -> None:
    run_id = str(manifest["run_id"])
    attempt_id = str(
        _attempt_payload(run_id, job, sample_index, "requested")["attempt_id"]
    )
    identity = {
        "run_id": run_id,
        "job_id": str(job["job_id"]),
        "attempt_id": attempt_id,
        "pocket_id": str(job["pocket_id"]),
        "seed": int(job["seed"]),
        "intervention": str(job["arm_id"]),
    }
    tracer = AttemptTracer(trace_writer, identity)
    runtime["seed_all"](int(job["seed"]) + sample_index)
    threshold = copy.deepcopy(runtime["policy"].sample.threshold)
    data = runtime["transform_data"](copy.deepcopy(base_data), runtime["masking"])
    ledger.append(_attempt_payload(run_id, job, sample_index, "initialized"))
    tracer.decision(0, "attempt.initialized", {"queue_size": 0})
    if diagnostics_enabled:
        _configure_diagnostics(
            runtime,
            data,
            job,
            0,
            tracer,
            wrong_data,
            pocket_record,
            wrong_record,
        )
    else:
        runtime["model"].set_diagnostics(None, None)
    ledger.append(_attempt_payload(run_id, job, sample_index, "sampling"))
    queue = runtime["get_init"](
        data.to(device), runtime["model"], runtime["composer"], threshold
    )
    queue = queue[: int(runtime["policy"].sample.beam_size)]
    tracer.decision(0, "queue.initialized", {"queue_size": len(queue)})
    initial_failure = classify_initial_queue(queue, runtime["STATUS_FINISHED"])
    if initial_failure is not None:
        _append_failure(
            ledger,
            run_id,
            job,
            sample_index,
            initial_failure,
            "initialization produced no runnable ligand candidate",
        )
        tracer.decision(0, "attempt.failed", {"reason": initial_failure})
        return
    max_steps = int(runtime["policy"].sample.max_steps)
    for step in range(1, max_steps + 1):
        queue_tmp = []
        queue_weights = []
        for parent in queue:
            if diagnostics_enabled:
                _configure_diagnostics(
                    runtime,
                    parent,
                    job,
                    step,
                    tracer,
                    wrong_data,
                    pocket_record,
                    wrong_record,
                )
            else:
                runtime["model"].set_diagnostics(None, None)
            candidates = runtime["get_next"](
                parent.to(device), runtime["model"], runtime["composer"], threshold
            )
            running = []
            for candidate in candidates:
                if candidate.status == runtime["STATUS_FINISHED"]:
                    ledger.append(_attempt_payload(run_id, job, sample_index, "generated"))
                    try:
                        molecule = runtime["reconstruct"](candidate)
                    except runtime["MolReconsError"] as exc:
                        _append_failure(
                            ledger, run_id, job, sample_index, "reconstruction_error", str(exc)
                        )
                        tracer.decision(step, "attempt.failed", {"reason": "reconstruction_error"})
                        return
                    smiles = runtime["Chem"].MolToSmiles(molecule)
                    if "." in smiles:
                        _append_failure(
                            ledger, run_id, job, sample_index, "disconnected", smiles
                        )
                        tracer.decision(step, "attempt.failed", {"reason": "disconnected"})
                        return
                    ledger.append(
                        _attempt_payload(
                            run_id,
                            job,
                            sample_index,
                            "reconstructed",
                            smiles=smiles,
                        )
                    )
                    duplicate = smiles in known_smiles
                    known_smiles.add(smiles)
                    candidate_path = job_root / "candidates" / f"{sample_index:04d}.pt"
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(candidate, candidate_path)
                    sdf_path = job_root / "sdf" / f"{sample_index:04d}.sdf"
                    sdf_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        _write_sdf(runtime["Chem"], molecule, sdf_path)
                    except Exception as exc:
                        ledger.append(
                            _attempt_payload(
                                run_id,
                                job,
                                sample_index,
                                "evaluated",
                                evaluation_status="failed",
                                error_code="sdf_write_error",
                                error_message=f"{type(exc).__name__}: {exc}"[:500],
                            )
                        )
                        tracer.decision(
                            step,
                            "attempt.failed",
                            {"reason": "sdf_write_error"},
                        )
                        return
                    ledger.append(
                        _attempt_payload(
                            run_id,
                            job,
                            sample_index,
                            "evaluated",
                            smiles=smiles,
                            duplicate=duplicate,
                            candidate_path=str(candidate_path.relative_to(run_root).as_posix()),
                            candidate_sha256=sha256_file(candidate_path),
                            sdf_path=str(sdf_path.relative_to(run_root).as_posix()),
                            sdf_sha256=sha256_file(sdf_path),
                        )
                    )
                    tracer.decision(
                        step,
                        "attempt.finished",
                        {"smiles": smiles, "duplicate": duplicate},
                    )
                    return
                if candidate.status == runtime["STATUS_RUNNING"]:
                    running.append(candidate)
            queue_tmp.extend(running)
            if running:
                queue_weights.extend([1.0 / len(running)] * len(running))
        if not queue_tmp:
            _append_failure(ledger, run_id, job, sample_index, "queue_empty", "queue exhausted")
            tracer.decision(step, "attempt.failed", {"reason": "queue_empty"})
            return
        probabilities = runtime["logp_to_rank_prob"](
            np.array([candidate.average_logp[2:] for candidate in queue_tmp]),
            queue_weights,
        )
        count = min(int(runtime["policy"].sample.beam_size), len(queue_tmp))
        selected = np.random.choice(
            np.arange(len(queue_tmp)), p=probabilities, size=count, replace=False
        )
        queue = [queue_tmp[index] for index in selected]
        tracer.decision(
            step,
            "queue.pruned",
            {"candidate_count": len(queue_tmp), "selected_indices": selected.tolist()},
        )
    _append_failure(ledger, run_id, job, sample_index, "max_steps", str(max_steps))
    tracer.decision(max_steps, "attempt.failed", {"reason": "max_steps"})


def run_job(
    manifest_path: Path,
    job_id: str,
    device: str,
    *,
    resume: bool = False,
) -> int:
    manifest_path = manifest_path.resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    job, job_root = load_declared_job(run_root, manifest, job_id)
    jobs_path = run_root / manifest["artifacts"]["jobs"]["path"]
    if sha256_file(jobs_path) != manifest["artifacts"]["jobs"]["sha256"]:
        raise ValueError("jobs.jsonl hash does not match the run manifest")
    attempts_path = job_root / "attempts.jsonl"
    events_path = job_root / "events.jsonl"
    if not resume:
        job_root.mkdir(parents=True, exist_ok=False)
        predeclare_attempts(attempts_path, str(manifest["run_id"]), job)
    elif not attempts_path.is_file() or not events_path.is_file():
        raise ValueError("resume requires existing attempts.jsonl and events.jsonl")

    replay = replay_ledger(attempts_path)
    if replay.truncated_final_line:
        raise ValueError("attempt ledger has a truncated final line")
    terminal = {
        attempt_id for attempt_id, status in replay.states.items() if status in {"evaluated", "failed"}
    }
    try:
        runtime = _load_runtime(manifest, device)
        pockets = _pocket_by_id(run_root, manifest)
        pocket_record = pockets[str(job["pocket_id"])]
        base_data = _build_pocket_data(runtime, pocket_record)
        wrong_record = None
        wrong_data = None
        if job["intervention"] == "D4":
            wrong_record = pockets[_d4_pair(run_root, str(job["pocket_id"]))]
            wrong_data = _build_pocket_data(runtime, wrong_record)
    except Exception as exc:
        with (job_root / "exceptions.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\n[bootstrap]\n{traceback.format_exc()}\n")
        close_requested_after_bootstrap_failure(
            attempts_path,
            str(manifest["run_id"]),
            job,
            f"{type(exc).__name__}: {exc}",
        )
        raise
    known_smiles = {
        str(record["smiles"])
        for record in replay.records
        if record.get("status") == "evaluated" and record.get("smiles")
    }
    log_path = job_root / "exceptions.log"
    with AttemptLedger(attempts_path, resume=True) as ledger, TraceWriter(
        events_path, resume=resume
    ) as trace_writer:
        for sample_index in range(int(job["attempt_count"])):
            attempt_id = AttemptRecord.new(
                str(manifest["run_id"]),
                str(job["pocket_id"]),
                int(job["seed"]),
                str(job["arm_id"]),
                sample_index,
            ).attempt_id
            if attempt_id in terminal:
                continue
            if resume and ledger.states.get(attempt_id) != "requested":
                recover_interrupted_attempt(
                    ledger,
                    str(manifest["run_id"]),
                    job,
                    sample_index,
                    job_root,
                )
                continue
            try:
                _run_attempt(
                    runtime,
                    run_root,
                    job_root,
                    manifest,
                    job,
                    sample_index,
                    ledger,
                    trace_writer,
                    pocket_record,
                    wrong_record,
                    base_data,
                    wrong_data,
                    known_smiles,
                    device,
                )
            except Exception as exc:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{attempt_id}]\n{traceback.format_exc()}\n")
                current = ledger.states.get(attempt_id)
                if current not in {"evaluated", "failed"}:
                    close_attempt_after_runtime_error(
                        ledger,
                        str(manifest["run_id"]),
                        job,
                        sample_index,
                        "runtime_error",
                        f"{type(exc).__name__}: {exc}",
                    )
            _atomic_checkpoint(
                job_root / "job-checkpoint.json",
                {
                    "schema_version": "phase0.v1",
                    "job_id": job_id,
                    "terminal_attempt_count": sum(
                        status in {"evaluated", "failed"}
                        for status in ledger.states.values()
                    ),
                },
            )
    return 0


def main() -> int:
    args = parse_args()
    return run_job(args.manifest, args.job_id, args.device, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
