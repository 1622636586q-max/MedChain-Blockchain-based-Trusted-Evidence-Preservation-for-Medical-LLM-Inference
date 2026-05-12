import copy
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.offchain_vault import (
    maybe_audit_datasets_path,
    project_root_from_records_dir,
    read_text_under_records,
    require_vault_auth,
    write_text_under_records,
)


def _infer_modality(image_path: str) -> str:
    p = (image_path or '').lower()
    if 'medpix2.0' in p or '/medpix' in p or 'mpx' in p:
        return 'medpix_general'
    if 'xray' in p or 'chest' in p or 'nih' in p:
        return 'cxr'
    if 'isic' in p or 'skin' in p or 'derm' in p:
        return 'derm'
    if 'appendix' in p or 'ultrasound' in p or 'us_' in p:
        return 'us'
    if 'ct' in p:
        return 'ct'
    if 'mri' in p:
        return 'mri'
    return 'unknown'


def _infer_anatomy_site(modality: str) -> str:
    if modality == 'medpix_general':
        return 'mixed_or_unknown'
    if modality == 'cxr':
        return 'chest'
    if modality == 'derm':
        return 'skin'
    if modality == 'us':
        return 'abdomen_appendix'
    return 'unknown'


@dataclass
class InferenceRequest:
    request_id: str
    patient_pseudonym: str
    study_uid: str
    image_digest: str
    prompt_digest: str
    model_name: str
    model_weights_sha256: str
    inference_config: str
    output_label: str
    output_confidence: float


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_canonical_payload(req: InferenceRequest) -> Dict[str, object]:
    return {
        "input_summary": {
            "patient_pseudonym": req.patient_pseudonym,
            "study_uid": req.study_uid,
            "image_digest": req.image_digest,
            "prompt_digest": req.prompt_digest,
        },
        "model_summary": {
            "model_name": req.model_name,
            "model_weights_sha256": req.model_weights_sha256,
            "inference_config": req.inference_config,
        },
        "output_summary": {
            "output_label": req.output_label,
            "output_confidence": round(req.output_confidence, 4),
        },
        "request_id": req.request_id,
    }


def compute_commitment(payload: Dict[str, object], salt: str) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return _sha256(canonical + "|" + salt)


def apply_payload_ablation(payload: Dict[str, object], mode: str) -> Dict[str, object]:
    """
    Ablate the proposed canonical_payload before commitment (paper ablation study).
    Baseline A/B are still derived from the same (ablated) proposed payload for fair comparison.
    """
    m = (mode or "full").strip().lower()
    if m == "full":
        return copy.deepcopy(payload)
    p = copy.deepcopy(payload)
    if m == "no_governance":
        p.pop("governance_summary", None)
        return p
    if m == "no_model":
        p.pop("model_summary", None)
        return p
    if m == "output_only":
        return {
            "request_id": p["request_id"],
            "output_summary": copy.deepcopy(p.get("output_summary", {})),
        }
    raise ValueError(f"Unsupported payload_ablation mode: {mode}")


def _commitment_salt_csv(salt: str, domain_separator: str) -> str:
    return salt + "|" + domain_separator


def build_baseline_a_payload(full_payload: Dict[str, object]) -> Dict[str, object]:
    """Baseline A: output-only binding (no full inference context)."""
    return {
        "request_id": full_payload["request_id"],
        "output_summary": copy.deepcopy(full_payload["output_summary"]),
    }


def build_baseline_b_payload(
    full_payload: Dict[str, object],
    pred_output: str,
    image_path: str,
) -> Dict[str, object]:
    """
    Baseline B: heavier on-chain style binding (near-raw proxy plus full summaries).
    Simulates anchoring more clinically detailed material on chain.
    """
    p = copy.deepcopy(full_payload)
    raw = (pred_output or "")[:2000] + "|" + (image_path or "")
    p["heavy_onchain_raw_digest"] = _sha256(raw)
    return p


def _payload_top_level_key_count(payload: Dict[str, object]) -> int:
    return len(payload.keys())


def _write_baseline_commitment_specs(
    records_dir: Path,
    sample_payload: Dict[str, object],
    sample_pred_output: str,
    sample_image_path: str,
) -> None:
    """
    Human-readable map of what each baseline commits vs what the mock ledger stores.
    Baseline A/B commitments are experiment artifacts (see experiment_baselines.csv).
    The mock chain ledger column commitments_chain.commitment_hash is the proposed method.
    """
    pay_a = build_baseline_a_payload(sample_payload)
    pay_b = build_baseline_b_payload(sample_payload, sample_pred_output, sample_image_path)
    specs = {
        "where_to_look": {
            "proposed_commitment_on_mock_ledger": "commitments_chain.csv column commitment_hash",
            "proposed_offchain_detail": "inference_offchain.jsonl field canonical_payload (+ source_record digests)",
            "baseline_a_b_commitments": "experiment_baselines.csv columns commitment_baseline_a / commitment_baseline_b",
            "per_row_audit_flags": "experiment_baselines.csv columns audit_match_baseline_a / audit_match_baseline_b / audit_match_proposed",
        },
        "baseline_a": {
            "description": "Output-only commitment: binds request_id + output_summary only.",
            "committed_object_top_level_keys": sorted(pay_a.keys()),
            "extra_raw_fields": [],
        },
        "baseline_b": {
            "description": "Heavy commitment proxy: full proposed canonical payload plus a near-raw digest field.",
            "committed_object_top_level_keys": sorted(pay_b.keys()),
            "extra_raw_fields": [
                {
                    "field": "heavy_onchain_raw_digest",
                    "definition": "sha256( pred_output[:2000] + '|' + image_path )",
                    "note": "This is a near-raw binding proxy; pred_output itself is not written into the commitment JSON.",
                }
            ],
        },
        "proposed": {
            "description": "Medical canonical commitment used as the primary mock-ledger anchor.",
            "committed_object_top_level_keys": sorted(sample_payload.keys()),
            "extra_raw_fields": [],
        },
    }
    (records_dir / "baseline_commitment_specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sum_tag_audit_artifact_bytes(records_dir: Path) -> int:
    """Byte size of standard audit artifacts under a tag folder (after writes)."""
    names = (
        "commitments_chain.csv",
        "inference_offchain.jsonl",
        "tamper_audit.csv",
        "experiment_baselines.csv",
        "experiment_summary.json",
        "baseline_commitment_specs.json",
    )
    total = 0
    for name in names:
        p = records_dir / name
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def build_performance_overhead(
    records_dir: Path,
    n: int,
    audit_total_ms: float,
    perf_t0: float,
    cpu_t0: float,
) -> dict[str, object]:
    """Wall vs CPU time, audit-only wall time, and on-disk IO volume (normalized per 1k records)."""
    perf_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    nn = max(int(n), 1)
    wall = perf_t1 - perf_t0
    cpu = cpu_t1 - cpu_t0
    tag_bytes = _sum_tag_audit_artifact_bytes(records_dir)
    return {
        "wall_seconds_total": round(wall, 4),
        "process_cpu_seconds_total": round(cpu, 4),
        "wall_seconds_per_1000_records": round(wall / nn * 1000.0, 6),
        "process_cpu_seconds_per_1000_records": round(cpu / nn * 1000.0, 6),
        "audit_wall_ms_total": round(float(audit_total_ms), 3),
        "audit_wall_ms_per_1000_records": round(float(audit_total_ms) / nn * 1000.0, 4),
        "tag_artifact_bytes_on_disk": int(tag_bytes),
        "tag_artifact_bytes_per_1000_records": round(tag_bytes / nn * 1000.0, 2),
        "notes": (
            "wall_* uses time.perf_counter (elapsed wall time, includes I/O wait). "
            "process_cpu_* uses time.process_time (CPU seconds attributed to this process). "
            "audit_* is wall time for the audit/hash loop only (not CSV ingest or JSONL write). "
            "tag_artifact_bytes sums listed CSV/JSON/JSONL under this tag's records_dir "
            "(excludes assumptions.json, run_guide.json, market_data.json). "
            "Symlink targets are counted as file size on POSIX; behavior may differ on Windows."
        ),
    }


def _input_csv_stat_fingerprint(input_csv: Path) -> str:
    """Byte size so the same path with a replaced/larger CSV gets a new dataset_key (mtime omitted for cross-host reproducibility)."""
    try:
        return str(input_csv.stat().st_size)
    except OSError:
        return "missing"


def _dataset_key_from_inputs(
    input_csv: Path,
    max_rows: int,
    model_name: str,
    model_weights_sha256: str,
    inference_config: str,
    policy_version: str,
    prompt_template_version: str,
    app_scene: str,
    domain_separator: str,
    payload_ablation: str = "full",
) -> str:
    # Use as_posix() (not resolve()) so the same logical dataset matches across machines.
    # Include csv stat so replacing the file under the same path invalidates stale _datasets/.
    raw = "|".join(
        [
            str(input_csv.as_posix()),
            str(max_rows),
            _input_csv_stat_fingerprint(input_csv),
            str(model_name),
            str(model_weights_sha256),
            str(inference_config),
            str(policy_version),
            str(prompt_template_version),
            str(app_scene),
            str(domain_separator),
            str(payload_ablation),
        ]
    )
    return _sha256(raw)[:24]


def _write_shared_ref(records_dir: Path, dataset_dir: Path, role: str) -> None:
    ref = {
        "records_layout": "split",
        "role": role,
        "shared_dataset_dir": str(dataset_dir),
        "shared_inference_offchain_jsonl": str(dataset_dir / "inference_offchain.jsonl"),
        "shared_commitments_chain_csv": str(dataset_dir / "commitments_chain.csv"),
    }
    (records_dir / "shared_dataset_ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_symlink(target: Path, link_path: Path) -> bool:
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(target, link_path)
        return True
    except OSError:
        return False


def resolve_records_chain_csv(records_dir: Path) -> Path:
    """Prefer tag-local file; otherwise follow shared_dataset_ref (split layout)."""
    direct = records_dir / "commitments_chain.csv"
    if direct.is_file() or direct.is_symlink():
        try:
            maybe_audit_datasets_path("resolve_chain_csv", direct.resolve(), records_dir)
        except OSError:
            pass
        return direct
    refp = records_dir / "shared_dataset_ref.json"
    if refp.is_file():
        data = json.loads(refp.read_text(encoding="utf-8"))
        p = Path(str(data.get("shared_commitments_chain_csv", "")))
        if p.is_file():
            maybe_audit_datasets_path("resolve_chain_csv", p.resolve(), records_dir)
            return p
    return direct


def resolve_records_inference_offchain(records_dir: Path) -> Path:
    direct = records_dir / "inference_offchain.jsonl"
    if direct.is_file() or direct.is_symlink():
        try:
            maybe_audit_datasets_path("resolve_offchain_jsonl", direct.resolve(), records_dir)
        except OSError:
            pass
        return direct
    refp = records_dir / "shared_dataset_ref.json"
    if refp.is_file():
        data = json.loads(refp.read_text(encoding="utf-8"))
        p = Path(str(data.get("shared_inference_offchain_jsonl", "")))
        if p.is_file():
            maybe_audit_datasets_path("resolve_offchain_jsonl", p.resolve(), records_dir)
            return p
    return direct


def _reprice_chain_rows(
    rows: List[Dict[str, object]],
    gas_price_gwei: float,
    eth_usd: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        nr = dict(r)
        gas_used = int(float(str(nr.get("gas_used", 0))))
        gas_low = int(float(str(nr.get("gas_used_low", gas_used))))
        gas_mid = int(float(str(nr.get("gas_used_mid", gas_used))))
        gas_high = int(float(str(nr.get("gas_used_high", gas_used))))
        gas_low, gas_mid, gas_high = sorted([gas_low, gas_mid, gas_high])
        fee_eth = (gas_used * gas_price_gwei) / 1e9
        fee_usd = fee_eth * eth_usd
        fee_low_usd = ((gas_low * gas_price_gwei) / 1e9) * eth_usd
        fee_mid_usd = ((gas_mid * gas_price_gwei) / 1e9) * eth_usd
        fee_high_usd = ((gas_high * gas_price_gwei) / 1e9) * eth_usd
        nr["gas_used_low"] = gas_low
        nr["gas_used_mid"] = gas_mid
        nr["gas_used_high"] = gas_high
        nr["gas_price_gwei"] = round(gas_price_gwei, 6)
        nr["fee_est_eth"] = round(fee_eth, 12)
        nr["fee_est_usd"] = round(fee_usd, 8)
        nr["fee_low_usd"] = round(fee_low_usd, 8)
        nr["fee_mid_usd"] = round(fee_mid_usd, 8)
        nr["fee_high_usd"] = round(fee_high_usd, 8)
        out.append(nr)
    return out


def _load_pending_from_shared_offchain(
    shared_jsonl: Path, default_domain_separator: str, records_dir: Path
) -> List[Dict[str, object]]:
    pending: List[Dict[str, object]] = []
    text = read_text_under_records(shared_jsonl, records_dir)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rid = str(rec["request_id"])
        payload_orig = rec["canonical_payload"]
        salt = str(rec["salt"])
        sr = rec.get("source_record", {}) or {}
        image_path = str(sr.get("image_path", ""))
        pred_label = str(sr.get("pred_label", ""))
        gold_label = str(sr.get("gold_label", ""))
        input_text = str(sr.get("input_text_preview", ""))
        pred_output = str(sr.get("pred_output_text_preview", ""))
        gov = (payload_orig.get("governance_summary", {}) if isinstance(payload_orig, dict) else {}) or {}
        dom = str(gov.get("domain_separator", "") or default_domain_separator)
        salt_bind = _commitment_salt_csv(salt, dom)
        pending.append(
            {
                "request_id": rid,
                "payload_orig": payload_orig,
                "salt": salt,
                "salt_bind": salt_bind,
                "pred_output": pred_output,
                "image_path": image_path,
                "pred_label": pred_label,
                "gold_label": gold_label,
                "input_text": input_text,
                "commitment_proposed": "",
                "chain_row": None,
            }
        )
    return pending


def _attach_chain_rows_to_pending(pending: List[Dict[str, object]], chain_rows: List[Dict[str, object]]) -> None:
    by_id = {str(r["request_id"]): r for r in chain_rows}
    for p in pending:
        rid = str(p["request_id"])
        row = by_id.get(rid)
        if row is None:
            raise ValueError(f"Missing chain row for request_id={rid} when loading shared dataset")
        p["chain_row"] = row
        p["commitment_proposed"] = str(row["commitment_hash"])


def _load_chain_rows_from_csv(chain_csv: Path, records_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    text = read_text_under_records(chain_csv, records_dir)
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rows.append(dict(row))
    return rows


def _tamper_output_label(payload: Dict[str, object]) -> None:
    out = payload.setdefault("output_summary", {})
    label = str(out.get("output_label") or "")
    out["output_label"] = (label + "[tamper]")[:500] if label else "tampered_label"


def _tamper_context_modality(payload: Dict[str, object]) -> bool:
    inp = payload.get("input_summary")
    if not isinstance(inp, dict):
        return False
    mod = str(inp.get("modality") or "unknown")
    inp["modality"] = mod + "_tampered"
    return True


def _tamper_model_inference_config(payload: Dict[str, object]) -> bool:
    ms = payload.get("model_summary")
    if not isinstance(ms, dict):
        return False
    ic = str(ms.get("inference_config", ""))
    ms["inference_config"] = (ic + "|tampered_model")[:800]
    return True


def _tamper_governance_policy_version(payload: Dict[str, object]) -> bool:
    gov = payload.get("governance_summary")
    if not isinstance(gov, dict):
        return False
    pv = str(gov.get("policy_version", ""))
    gov["policy_version"] = (pv + "_tampered_gov")[:200]
    return True


def _tamper_field_swap(payload: Dict[str, object]) -> None:
    """Swap model_summary and governance_summary to simulate field mis-binding."""
    if "model_summary" not in payload or "governance_summary" not in payload:
        return
    a = payload["model_summary"]
    b = payload["governance_summary"]
    payload["model_summary"] = copy.deepcopy(b)
    payload["governance_summary"] = copy.deepcopy(a)


def _tamper_truncate_partial(payload: Dict[str, object]) -> None:
    """Truncate output_label (and high-signal digests) — structural truncation attack."""
    out = payload.get("output_summary")
    if isinstance(out, dict):
        lab = str(out.get("output_label") or "")
        if len(lab) > 2:
            out["output_label"] = lab[: max(1, len(lab) // 2)]
        elif lab:
            out["output_label"] = lab[:1]
        pd = out.get("pred_output_digest")
        if isinstance(pd, str) and len(pd) > 12:
            out["pred_output_digest"] = pd[:12]


def _tamper_cross_sample_splice(
    payload: Dict[str, object], mixed_index: int, pending: Optional[List[Dict[str, object]]]
) -> None:
    """Replace output_summary with another row's (deterministic peer index)."""
    if not pending or len(pending) < 2:
        return
    n = len(pending)
    peer_idx = (mixed_index + 17) % n
    if peer_idx == mixed_index:
        peer_idx = (mixed_index + 1) % n
    peer_pl = pending[peer_idx].get("payload_orig")
    if isinstance(peer_pl, dict) and isinstance(peer_pl.get("output_summary"), dict):
        payload["output_summary"] = copy.deepcopy(peer_pl["output_summary"])


def _tamper_layer_quarter(payload: Dict[str, object], mixed_index: int) -> None:
    """
    Exactly one logical layer per row, chosen by row index (no cascade).

    Cycles model -> governance -> context -> output. If the chosen block is absent
    (e.g. model_summary removed under no_model), the mutation is a no-op for that row,
    so proposed tamper-detection rates step with payload_ablation (~1.0 / ~0.75 / ~0.25
    for full / no_model / output_only at large n).
    """
    k = mixed_index % 4
    if k == 0:
        _tamper_model_inference_config(payload)
    elif k == 1:
        _tamper_governance_policy_version(payload)
    elif k == 2:
        _tamper_context_modality(payload)
    else:
        _tamper_output_label(payload)


def _tamper_offchain_payload(
    payload: Dict[str, object],
    strategy: str,
    mixed_index: int,
    *,
    pending: Optional[List[Dict[str, object]]] = None,
) -> None:
    """In-place mutation of canonical payload to simulate off-chain tampering."""
    if strategy == "mixed":
        strategy = "output" if mixed_index % 2 == 0 else "context"

    if strategy == "salt_corrupt":
        # Salt binding is corrupted in the audit loop, not in JSON payload.
        return

    if strategy == "field_swap":
        _tamper_field_swap(payload)
        return

    if strategy == "truncate_partial":
        _tamper_truncate_partial(payload)
        return

    if strategy == "cross_sample_splice":
        _tamper_cross_sample_splice(payload, mixed_index, pending)
        return

    if strategy == "layer_quarter":
        _tamper_layer_quarter(payload, mixed_index)
        return

    if strategy == "model":
        # Cascade so ablations do not all collapse to the same output-only mutation.
        if _tamper_model_inference_config(payload):
            return
        if _tamper_governance_policy_version(payload):
            return
        if _tamper_context_modality(payload):
            return
        _tamper_output_label(payload)
        return

    if strategy == "governance":
        if _tamper_governance_policy_version(payload):
            return
        if _tamper_context_modality(payload):
            return
        _tamper_output_label(payload)
        return

    if strategy == "context":
        if _tamper_context_modality(payload):
            return
        _tamper_output_label(payload)
        return

    # "output" and any unknown strategy
    _tamper_output_label(payload)


def _mock_requests(num_requests: int) -> List[InferenceRequest]:
    labels = ["normal", "pneumonia", "nodule", "effusion", "other"]
    reqs: List[InferenceRequest] = []
    for _ in range(num_requests):
        rid = str(uuid.uuid4())
        conf = random.uniform(0.51, 0.99)
        reqs.append(
            InferenceRequest(
                request_id=rid,
                patient_pseudonym=f"pt_{random.randint(1000, 9999)}",
                study_uid=f"study_{random.randint(100000, 999999)}",
                image_digest=_sha256(str(uuid.uuid4())),
                prompt_digest=_sha256("diagnose chest xray"),
                model_name="qwen2.5-vl-lora-med",
                model_weights_sha256=_sha256("weights_v1.3"),
                inference_config="temperature=0.1,max_tokens=128",
                output_label=random.choice(labels),
                output_confidence=conf,
            )
        )
    return reqs


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _short_text(text: str, max_len: int = 120) -> str:
    text = text.strip().replace("\r", " ").replace("\n", " ")
    return text[:max_len] if len(text) > max_len else text


def _pick_row_value(row: Dict[str, object], candidates: List[str], default: str = "") -> str:
    """
    Read value from CSV row using multiple candidate headers.
    Handles UTF-8 BOM in first-column header and common alias names.
    """
    for key in candidates:
        if key in row and row.get(key) is not None:
            val = str(row.get(key, "")).strip()
            if val:
                return val
    # BOM fallback: first column may be like '\ufeffimage_path'
    for raw_key in row.keys():
        if raw_key.lstrip("\ufeff") in candidates:
            val = str(row.get(raw_key, "")).strip()
            if val:
                return val
    return default


def run_pipeline(records_dir: Path, num_requests: int = 100, tamper_rate: float = 0.08) -> None:
    chain_csv = records_dir / "commitments_chain.csv"
    offchain_jsonl = records_dir / "inference_offchain.jsonl"
    audit_csv = records_dir / "tamper_audit.csv"

    reqs = _mock_requests(num_requests)
    chain_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    with offchain_jsonl.open("w", encoding="utf-8") as off_f:
        for req in reqs:
            payload = build_canonical_payload(req)
            salt = _sha256(req.request_id)[:16]
            commitment_hash = compute_commitment(payload, salt)

            t0 = time.perf_counter()
            simulated_chain_latency_ms = random.uniform(90, 420)
            time.sleep(simulated_chain_latency_ms / 10000.0)
            gas_used = random.randint(42000, 82000)
            tx_hash = _sha256(req.request_id + commitment_hash)[:64]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            chain_rows.append(
                {
                    "request_id": req.request_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "commitment_hash": commitment_hash,
                    "model_weights_sha256": req.model_weights_sha256,
                    "tx_hash": tx_hash,
                    "gas_used": gas_used,
                    "write_latency_ms": round(elapsed_ms, 2),
                }
            )

            offchain_record = {
                "request_id": req.request_id,
                "salt": salt,
                "canonical_payload": payload,
                "encrypted_blob_ref": f"s3://secure-bucket/{req.request_id}.bin",
            }
            off_f.write(json.dumps(offchain_record, ensure_ascii=False) + "\n")

    # Simulate tampering according to tamper_rate for attack scenario analysis.
    tamper_count = int(round(num_requests * tamper_rate))
    tamper_count = min(max(tamper_count, 0), num_requests)
    tampered_ids = set(random.sample([r.request_id for r in reqs], k=tamper_count))
    for row in chain_rows:
        req_id = row["request_id"]
        matched = req_id not in tampered_ids
        audit_rows.append(
            {
                "request_id": req_id,
                "chain_commitment_hash": row["commitment_hash"],
                "recomputed_hash": row["commitment_hash"] if matched else _sha256(str(uuid.uuid4())),
                "is_match": int(matched),
            }
        )

    with chain_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "timestamp_utc",
                "commitment_hash",
                "model_weights_sha256",
                "tx_hash",
                "gas_used",
                "write_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(chain_rows)

    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "chain_commitment_hash", "recomputed_hash", "is_match"])
        writer.writeheader()
        writer.writerows(audit_rows)


def run_pipeline_from_csv(
    input_csv: Path,
    records_dir: Path,
    model_name: str,
    model_weights_sha256: str,
    inference_config: str = "temperature=0.1,max_tokens=128",
    tamper_rate: float = 0.0,
    max_rows: int = 0,
    simulate_chain_delay: bool = False,
    gas_used_fixed: int = 62000,
    gas_used_range: List[int] | None = None,
    gas_price_gwei: float = 0.10,
    eth_usd: float = 3000.0,
    policy_version: str = "med_policy_v1",
    prompt_template_version: str = "prompt_tpl_v1",
    app_scene: str = "medical_diagnosis",
    domain_separator: str = "MED-AUDIT-V1",
    tamper_strategy: str = "output",
    write_baseline_experiment: bool = True,
    records_layout: str = "flat",
    pipeline_role: str = "full",
    shared_dataset_dir: Path | None = None,
    payload_ablation: str = "full",
    random_seed: int | None = None,
) -> None:
    records_layout = (records_layout or "flat").strip().lower()
    pipeline_role = (pipeline_role or "full").strip().lower()
    if records_layout not in ("flat", "split"):
        raise ValueError(f"records_layout must be flat|split, got {records_layout}")
    if records_layout == "flat" and pipeline_role != "full":
        raise ValueError("flat records_layout only supports pipeline_role=full")
    if records_layout == "split" and pipeline_role not in ("full", "tamper_from_shared", "cost_from_shared"):
        raise ValueError(
            "split records_layout requires pipeline_role in full|tamper_from_shared|cost_from_shared"
        )

    records_dir.mkdir(parents=True, exist_ok=True)
    require_vault_auth(project_root_from_records_dir(records_dir))
    _perf_t0 = time.perf_counter()
    _cpu_t0 = time.process_time()
    audit_total_ms = 0.0
    chain_csv = records_dir / "commitments_chain.csv"
    offchain_jsonl = records_dir / "inference_offchain.jsonl"
    audit_csv = records_dir / "tamper_audit.csv"

    dataset_dir_for_ref: Path | None = None
    meta_shared_dataset_dir = str(shared_dataset_dir) if shared_dataset_dir else ""
    payload_ablation = (payload_ablation or "full").strip().lower()

    if records_layout == "split" and pipeline_role == "cost_from_shared":
        if shared_dataset_dir is None:
            raise ValueError("shared_dataset_dir is required when pipeline_role=cost_from_shared")
        sd_cost = shared_dataset_dir.resolve()
        shared_chain = sd_cost / "commitments_chain.csv"
        maybe_audit_datasets_path("read_shared_chain_cost_branch", sd_cost, records_dir)
        if not shared_chain.is_file():
            raise FileNotFoundError(f"Missing shared chain CSV: {shared_chain}")
        repriced = _reprice_chain_rows(
            _load_chain_rows_from_csv(shared_chain, records_dir),
            gas_price_gwei=gas_price_gwei,
            eth_usd=eth_usd,
        )
        _write_shared_ref(records_dir, sd_cost, "cost")
        shared_off = sd_cost / "inference_offchain.jsonl"
        if shared_off.is_file():
            _try_symlink(shared_off, records_dir / "inference_offchain.jsonl")
        fieldnames = [
            "request_id",
            "timestamp_utc",
            "commitment_hash",
            "model_weights_sha256",
            "tx_hash",
            "gas_used",
            "gas_used_low",
            "gas_used_mid",
            "gas_used_high",
            "gas_price_gwei",
            "fee_est_eth",
            "fee_est_usd",
            "fee_low_usd",
            "fee_mid_usd",
            "fee_high_usd",
            "write_latency_ms",
        ]
        with chain_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(repriced)
        assumptions = {
            "note": "cost_from_shared: recomputed fee columns from shared canonical chain; gas_used unchanged.",
            "records_layout": "split",
            "pipeline_role": "cost_from_shared",
            "payload_ablation": payload_ablation,
            "shared_dataset_dir": str(sd_cost),
            "gas_used_fixed": int(gas_used_fixed),
            "gas_used_range": [int(x) for x in gas_used_range] if gas_used_range else [int(gas_used_fixed)] * 3,
            "gas_price_gwei": float(gas_price_gwei),
            "eth_usd": float(eth_usd),
            "policy_version": str(policy_version),
            "prompt_template_version": str(prompt_template_version),
            "app_scene": str(app_scene),
            "domain_separator": str(domain_separator),
            "simulate_chain_delay": bool(simulate_chain_delay),
            "tamper_rate": float(tamper_rate),
            "tamper_strategy": str(tamper_strategy),
            "write_baseline_experiment": bool(write_baseline_experiment),
            "input_csv": str(input_csv),
        }
        (records_dir / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    pending: List[Dict[str, object]] = []

    if records_layout == "split" and pipeline_role == "tamper_from_shared":
        if shared_dataset_dir is None:
            raise ValueError("shared_dataset_dir is required when pipeline_role=tamper_from_shared")
        sd_tamper = shared_dataset_dir.resolve()
        sj = sd_tamper / "inference_offchain.jsonl"
        sc = sd_tamper / "commitments_chain.csv"
        maybe_audit_datasets_path("read_shared_tamper_branch", sd_tamper, records_dir)
        if not sj.is_file() or not sc.is_file():
            raise FileNotFoundError(f"Missing shared dataset files under {sd_tamper}")
        pending = _load_pending_from_shared_offchain(sj, domain_separator, records_dir)
        _attach_chain_rows_to_pending(pending, _load_chain_rows_from_csv(sc, records_dir))
        _write_shared_ref(records_dir, sd_tamper, "tamper")
        meta_shared_dataset_dir = str(sd_tamper)
        _try_symlink(sc, chain_csv)
        offchain_jsonl = records_dir / "inference_offchain.jsonl"
    elif records_layout == "split" and pipeline_role == "full":
        if tamper_rate > 0:
            raise ValueError(
                "split layout pipeline_role=full builds the canonical dataset only (tamper_rate must be 0). "
                "Use tamper_from_shared for tampered runs."
            )
        dk = _dataset_key_from_inputs(
            input_csv,
            max_rows,
            model_name,
            model_weights_sha256,
            inference_config,
            policy_version,
            prompt_template_version,
            app_scene,
            domain_separator,
            payload_ablation,
        )
        dataset_dir_for_ref = (records_dir.parent / "_datasets" / dk).resolve()
        dataset_dir_for_ref.mkdir(parents=True, exist_ok=True)
        maybe_audit_datasets_path("write_prepare_canonical_dataset_dir", dataset_dir_for_ref, records_dir)
        chain_csv = dataset_dir_for_ref / "commitments_chain.csv"
        offchain_jsonl = dataset_dir_for_ref / "inference_offchain.jsonl"
        with input_csv.open("r", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            for idx, row in enumerate(reader):
                if max_rows > 0 and idx >= max_rows:
                    break
                image_path = _pick_row_value(row, ["image_path", "img_path", "path"], default="")
                pred_label = _pick_row_value(row, ["pred_label", "pred"], default="")
                pred_prob = _safe_float(
                    _pick_row_value(row, ["pred_prob", "prob", "confidence"], default="0.0"),
                    default=0.0,
                )
                topk_json = _pick_row_value(row, ["topk_json", "topk", "top_k_json"], default="[]")
                gold_label = _pick_row_value(row, ["gold_label", "label", "gold"], default="")
                input_text = _pick_row_value(row, ["input", "prompt", "question"], default="")
                pred_output = _pick_row_value(row, ["pred_output", "prediction", "output_pred"], default="")
                gold_output = _pick_row_value(row, ["gold_output", "target", "output_gold"], default="")

                if not pred_label:
                    pred_label = _short_text(pred_output, max_len=80) or "unknown"
                if not gold_label:
                    gold_label = _short_text(gold_output, max_len=80)

                request_id = str(uuid.uuid4())
                modality = _infer_modality(image_path)
                anatomy_site = _infer_anatomy_site(modality)
                payload: Dict[str, object] = {
                    "input_summary": {
                        "patient_pseudonym": _sha256(image_path)[:12],
                        "encounter_digest": _sha256(image_path + "|encounter")[:20],
                        "study_uid": _sha256(image_path + "|study")[:16],
                        "image_digest": _sha256(image_path),
                        "prompt_digest": _sha256(input_text if input_text else "medical_multimodal_diagnosis"),
                        "prompt_template_version": prompt_template_version,
                        "modality": modality,
                        "anatomy_site": anatomy_site,
                    },
                    "model_summary": {
                        "model_name": model_name,
                        "model_weights_sha256": model_weights_sha256,
                        "inference_config": inference_config,
                    },
                    "output_summary": {
                        "output_label": pred_label,
                        "output_confidence": round(pred_prob, 4),
                        "topk_digest": _sha256(topk_json),
                        "pred_output_digest": _sha256(pred_output) if pred_output else "",
                        "gold_label_digest": _sha256(gold_label) if gold_label else "",
                        "gold_output_digest": _sha256(gold_output) if gold_output else "",
                    },
                    "governance_summary": {
                        "app_scene": app_scene,
                        "policy_version": policy_version,
                        "domain_separator": domain_separator,
                    },
                    "request_id": request_id,
                }
                payload = apply_payload_ablation(payload, payload_ablation)
                salt = _sha256(request_id)[:16]
                salt_bind = _commitment_salt_csv(salt, domain_separator)
                commitment_hash = compute_commitment(payload, salt_bind)

                t0 = time.perf_counter()
                simulated_chain_latency_ms = random.uniform(90, 420)
                if simulate_chain_delay:
                    time.sleep(simulated_chain_latency_ms / 10000.0)
                gas_used = int(gas_used_fixed)
                gas_range = gas_used_range if gas_used_range else [gas_used, gas_used, gas_used]
                gas_low, gas_mid, gas_high = sorted([int(gas_range[0]), int(gas_range[1]), int(gas_range[2])])
                tx_hash = _sha256(request_id + commitment_hash)[:64]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                fee_eth = (gas_used * gas_price_gwei) / 1e9
                fee_usd = fee_eth * eth_usd
                fee_low_usd = ((gas_low * gas_price_gwei) / 1e9) * eth_usd
                fee_mid_usd = ((gas_mid * gas_price_gwei) / 1e9) * eth_usd
                fee_high_usd = ((gas_high * gas_price_gwei) / 1e9) * eth_usd

                pending.append(
                    {
                        "request_id": request_id,
                        "payload_orig": payload,
                        "salt": salt,
                        "salt_bind": salt_bind,
                        "pred_output": pred_output,
                        "image_path": image_path,
                        "pred_label": pred_label,
                        "gold_label": gold_label,
                        "input_text": input_text,
                        "commitment_proposed": commitment_hash,
                        "chain_row": {
                            "request_id": request_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "commitment_hash": commitment_hash,
                            "model_weights_sha256": model_weights_sha256,
                            "tx_hash": tx_hash,
                            "gas_used": gas_used,
                            "gas_used_low": gas_low,
                            "gas_used_mid": gas_mid,
                            "gas_used_high": gas_high,
                            "gas_price_gwei": round(gas_price_gwei, 6),
                            "fee_est_eth": round(fee_eth, 12),
                            "fee_est_usd": round(fee_usd, 8),
                            "fee_low_usd": round(fee_low_usd, 8),
                            "fee_mid_usd": round(fee_mid_usd, 8),
                            "fee_high_usd": round(fee_high_usd, 8),
                            "write_latency_ms": round(elapsed_ms, 2),
                        },
                    }
                )
    else:
        with input_csv.open("r", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            for idx, row in enumerate(reader):
                if max_rows > 0 and idx >= max_rows:
                    break
                image_path = _pick_row_value(row, ["image_path", "img_path", "path"], default="")
                pred_label = _pick_row_value(row, ["pred_label", "pred"], default="")
                pred_prob = _safe_float(
                    _pick_row_value(row, ["pred_prob", "prob", "confidence"], default="0.0"),
                    default=0.0,
                )
                topk_json = _pick_row_value(row, ["topk_json", "topk", "top_k_json"], default="[]")
                gold_label = _pick_row_value(row, ["gold_label", "label", "gold"], default="")
                input_text = _pick_row_value(row, ["input", "prompt", "question"], default="")
                pred_output = _pick_row_value(row, ["pred_output", "prediction", "output_pred"], default="")
                gold_output = _pick_row_value(row, ["gold_output", "target", "output_gold"], default="")

                if not pred_label:
                    pred_label = _short_text(pred_output, max_len=80) or "unknown"
                if not gold_label:
                    gold_label = _short_text(gold_output, max_len=80)

                request_id = str(uuid.uuid4())
                modality = _infer_modality(image_path)
                anatomy_site = _infer_anatomy_site(modality)
                payload = {
                    "input_summary": {
                        "patient_pseudonym": _sha256(image_path)[:12],
                        "encounter_digest": _sha256(image_path + "|encounter")[:20],
                        "study_uid": _sha256(image_path + "|study")[:16],
                        "image_digest": _sha256(image_path),
                        "prompt_digest": _sha256(input_text if input_text else "medical_multimodal_diagnosis"),
                        "prompt_template_version": prompt_template_version,
                        "modality": modality,
                        "anatomy_site": anatomy_site,
                    },
                    "model_summary": {
                        "model_name": model_name,
                        "model_weights_sha256": model_weights_sha256,
                        "inference_config": inference_config,
                    },
                    "output_summary": {
                        "output_label": pred_label,
                        "output_confidence": round(pred_prob, 4),
                        "topk_digest": _sha256(topk_json),
                        "pred_output_digest": _sha256(pred_output) if pred_output else "",
                        "gold_label_digest": _sha256(gold_label) if gold_label else "",
                        "gold_output_digest": _sha256(gold_output) if gold_output else "",
                    },
                    "governance_summary": {
                        "app_scene": app_scene,
                        "policy_version": policy_version,
                        "domain_separator": domain_separator,
                    },
                    "request_id": request_id,
                }
                payload = apply_payload_ablation(payload, payload_ablation)
                salt = _sha256(request_id)[:16]
                salt_bind = _commitment_salt_csv(salt, domain_separator)
                commitment_hash = compute_commitment(payload, salt_bind)

                t0 = time.perf_counter()
                simulated_chain_latency_ms = random.uniform(90, 420)
                if simulate_chain_delay:
                    time.sleep(simulated_chain_latency_ms / 10000.0)
                gas_used = int(gas_used_fixed)
                gas_range = gas_used_range if gas_used_range else [gas_used, gas_used, gas_used]
                gas_low, gas_mid, gas_high = sorted([int(gas_range[0]), int(gas_range[1]), int(gas_range[2])])
                tx_hash = _sha256(request_id + commitment_hash)[:64]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                fee_eth = (gas_used * gas_price_gwei) / 1e9
                fee_usd = fee_eth * eth_usd
                fee_low_usd = ((gas_low * gas_price_gwei) / 1e9) * eth_usd
                fee_mid_usd = ((gas_mid * gas_price_gwei) / 1e9) * eth_usd
                fee_high_usd = ((gas_high * gas_price_gwei) / 1e9) * eth_usd

                pending.append(
                    {
                        "request_id": request_id,
                        "payload_orig": payload,
                        "salt": salt,
                        "salt_bind": salt_bind,
                        "pred_output": pred_output,
                        "image_path": image_path,
                        "pred_label": pred_label,
                        "gold_label": gold_label,
                        "input_text": input_text,
                        "commitment_proposed": commitment_hash,
                        "chain_row": {
                            "request_id": request_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "commitment_hash": commitment_hash,
                            "model_weights_sha256": model_weights_sha256,
                            "tx_hash": tx_hash,
                            "gas_used": gas_used,
                            "gas_used_low": gas_low,
                            "gas_used_mid": gas_mid,
                            "gas_used_high": gas_high,
                            "gas_price_gwei": round(gas_price_gwei, 6),
                            "fee_est_eth": round(fee_eth, 12),
                            "fee_est_usd": round(fee_usd, 8),
                            "fee_low_usd": round(fee_low_usd, 8),
                            "fee_mid_usd": round(fee_mid_usd, 8),
                            "fee_high_usd": round(fee_high_usd, 8),
                            "write_latency_ms": round(elapsed_ms, 2),
                        },
                    }
                )

    request_ids = [str(p["request_id"]) for p in pending]
    n = len(request_ids)
    print(
        f"[audit_pipeline] row_count_this_run={n} pipeline_role={pipeline_role} "
        f"records_layout={records_layout} max_rows={max_rows} input_csv={input_csv}",
        file=sys.stderr,
        flush=True,
    )
    tamper_count = int(round(n * tamper_rate))
    tamper_count = min(max(tamper_count, 0), n)
    if random_seed is not None:
        random.seed(int(random_seed))
    tampered_ids: set[str] = set(random.sample(request_ids, k=tamper_count)) if tamper_count > 0 else set()

    chain_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    experiment_rows: List[Dict[str, object]] = []

    t_audit0 = time.perf_counter()
    for idx, p in enumerate(pending):
        req_id = str(p["request_id"])
        payload_orig: Dict[str, object] = p["payload_orig"]  # type: ignore[assignment]
        salt_bind = str(p["salt_bind"])
        salt = str(p["salt"])
        pred_output = str(p["pred_output"])
        image_path = str(p["image_path"])

        payload_off = copy.deepcopy(payload_orig)
        was_tampered = req_id in tampered_ids
        if was_tampered:
            _tamper_offchain_payload(payload_off, tamper_strategy, idx, pending=pending)

        salt_bind_audit = salt_bind
        if was_tampered and str(tamper_strategy) == "salt_corrupt":
            salt_bind_audit = salt_bind + "|wrong_salt_bind"

        chain_rows.append(p["chain_row"])  # type: ignore[arg-type]

        recomputed_proposed = compute_commitment(payload_off, salt_bind_audit)
        is_match = int(recomputed_proposed == p["commitment_proposed"])
        audit_rows.append(
            {
                "request_id": req_id,
                "chain_commitment_hash": p["commitment_proposed"],
                "recomputed_hash": recomputed_proposed,
                "is_match": is_match,
            }
        )

        if write_baseline_experiment:
            pay_a_orig = build_baseline_a_payload(payload_orig)
            pay_b_orig = build_baseline_b_payload(payload_orig, pred_output, image_path)
            comm_a = compute_commitment(pay_a_orig, salt_bind)
            comm_b = compute_commitment(pay_b_orig, salt_bind)
            pay_a_off = build_baseline_a_payload(payload_off)
            pay_b_off = build_baseline_b_payload(payload_off, pred_output, image_path)
            match_a = int(compute_commitment(pay_a_off, salt_bind_audit) == comm_a)
            match_b = int(compute_commitment(pay_b_off, salt_bind_audit) == comm_b)
            experiment_rows.append(
                {
                    "request_id": req_id,
                    "tampered": int(was_tampered),
                    "tamper_strategy": tamper_strategy,
                    "commitment_baseline_a": comm_a,
                    "commitment_baseline_b": comm_b,
                    "commitment_proposed": p["commitment_proposed"],
                    "audit_match_baseline_a": match_a,
                    "audit_match_baseline_b": match_b,
                    "audit_match_proposed": is_match,
                }
            )

    audit_total_ms = (time.perf_counter() - t_audit0) * 1000.0

    off_lines: List[str] = []
    for idx, p in enumerate(pending):
        req_id = str(p["request_id"])
        payload_orig = p["payload_orig"]  # type: ignore[assignment]
        payload_off = copy.deepcopy(payload_orig)
        if req_id in tampered_ids:
            _tamper_offchain_payload(payload_off, tamper_strategy, idx, pending=pending)
        salt = str(p["salt"])
        pred_output = str(p["pred_output"])
        image_path = str(p["image_path"])
        offchain_record = {
            "request_id": req_id,
            "salt": salt,
            "canonical_payload": payload_off,
            "encrypted_blob_ref": f"s3://secure-bucket/{req_id}.bin",
            "source_record": {
                "image_path": image_path,
                "pred_label": str(p["pred_label"]),
                "gold_label": str(p["gold_label"]),
                "input_text_preview": _short_text(str(p["input_text"]), max_len=500),
                "pred_output_text_preview": _short_text(pred_output, max_len=2000),
                "input_digest": _sha256(str(p["input_text"])) if p["input_text"] else "",
                "pred_output_digest": _sha256(pred_output) if pred_output else "",
            },
        }
        off_lines.append(json.dumps(offchain_record, ensure_ascii=False) + "\n")
    write_text_under_records(offchain_jsonl, "".join(off_lines), records_dir)

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "request_id",
            "timestamp_utc",
            "commitment_hash",
            "model_weights_sha256",
            "tx_hash",
            "gas_used",
            "gas_used_low",
            "gas_used_mid",
            "gas_used_high",
            "gas_price_gwei",
            "fee_est_eth",
            "fee_est_usd",
            "fee_low_usd",
            "fee_mid_usd",
            "fee_high_usd",
            "write_latency_ms",
        ],
    )
    writer.writeheader()
    writer.writerows(chain_rows)
    write_text_under_records(chain_csv, buf.getvalue(), records_dir)

    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "chain_commitment_hash", "recomputed_hash", "is_match"])
        writer.writeheader()
        writer.writerows(audit_rows)

    if write_baseline_experiment and experiment_rows and n > 0:
        exp_csv = records_dir / "experiment_baselines.csv"
        with exp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "request_id",
                    "tampered",
                    "tamper_strategy",
                    "commitment_baseline_a",
                    "commitment_baseline_b",
                    "commitment_proposed",
                    "audit_match_baseline_a",
                    "audit_match_baseline_b",
                    "audit_match_proposed",
                ],
            )
            w.writeheader()
            w.writerows(experiment_rows)

        benign = [r for r in experiment_rows if int(r["tampered"]) == 0]
        attacked = [r for r in experiment_rows if int(r["tampered"]) == 1]
        n_benign = len(benign)
        n_attacked = len(attacked)

        def _detected(rows: List[Dict[str, object]], key: str) -> int:
            return sum(1 for r in rows if int(r[key]) == 0)

        def _false_alarm(rows: List[Dict[str, object]], key: str) -> int:
            return sum(1 for r in rows if int(r[key]) == 0)

        summary = {
            "n_records": n,
            "tamper_rate": float(tamper_rate),
            "tamper_strategy": tamper_strategy,
            "payload_ablation": payload_ablation,
            "random_seed": random_seed,
            "tampered_count": tamper_count,
            "audit_total_ms": round(audit_total_ms, 3),
            "audit_per_record_ms": round(audit_total_ms / n, 6) if n else 0.0,
            "reproducibility_pass_rate_proposed": sum(int(r["audit_match_proposed"]) for r in experiment_rows) / n
            if n
            else 0.0,
            "tamper_detection_rate_proposed": (_detected(attacked, "audit_match_proposed") / n_attacked)
            if n_attacked
            else None,
            "false_alarm_rate_proposed": (_false_alarm(benign, "audit_match_proposed") / n_benign) if n_benign else 0.0,
            "baseline_a": {
                "on_chain_top_level_field_count": _payload_top_level_key_count(
                    build_baseline_a_payload(pending[0]["payload_orig"])  # type: ignore[arg-type]
                )
                if pending
                else 0,
                "tamper_detection_rate": (_detected(attacked, "audit_match_baseline_a") / n_attacked)
                if n_attacked
                else None,
                "false_alarm_rate": (_false_alarm(benign, "audit_match_baseline_a") / n_benign) if n_benign else 0.0,
                "reproducibility_pass_rate": sum(int(r["audit_match_baseline_a"]) for r in experiment_rows) / n
                if n
                else 0.0,
            },
            "baseline_b": {
                "on_chain_top_level_field_count": _payload_top_level_key_count(
                    build_baseline_b_payload(
                        pending[0]["payload_orig"],  # type: ignore[arg-type]
                        str(pending[0]["pred_output"]),
                        str(pending[0]["image_path"]),
                    )
                )
                if pending
                else 0,
                "tamper_detection_rate": (_detected(attacked, "audit_match_baseline_b") / n_attacked)
                if n_attacked
                else None,
                "false_alarm_rate": (_false_alarm(benign, "audit_match_baseline_b") / n_benign) if n_benign else 0.0,
                "reproducibility_pass_rate": sum(int(r["audit_match_baseline_b"]) for r in experiment_rows) / n
                if n
                else 0.0,
            },
            "proposed": {
                "on_chain_top_level_field_count": _payload_top_level_key_count(
                    pending[0]["payload_orig"]  # type: ignore[arg-type]
                )
                if pending
                else 0,
            },
            "cost_usd_mid_total": round(sum(float(r["fee_mid_usd"]) for r in chain_rows), 6),
            "cost_usd_low_total": round(sum(float(r["fee_low_usd"]) for r in chain_rows), 6),
            "cost_usd_high_total": round(sum(float(r["fee_high_usd"]) for r in chain_rows), 6),
            "privacy_proxy": {
                "baseline_a_committed_json_bytes": len(
                    json.dumps(
                        build_baseline_a_payload(pending[0]["payload_orig"]),  # type: ignore[arg-type]
                        sort_keys=True,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                )
                if pending
                else 0,
                "baseline_b_committed_json_bytes": len(
                    json.dumps(
                        build_baseline_b_payload(
                            pending[0]["payload_orig"],  # type: ignore[arg-type]
                            str(pending[0]["pred_output"]),
                            str(pending[0]["image_path"]),
                        ),
                        sort_keys=True,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                )
                if pending
                else 0,
                "proposed_committed_json_bytes": len(
                    json.dumps(
                        pending[0]["payload_orig"],  # type: ignore[arg-type]
                        sort_keys=True,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                )
                if pending
                else 0,
            },
            "notes": {
                "baseline_a": "output_summary only in commitment",
                "baseline_b": "full canonical plus heavy_onchain_raw_digest from truncated pred_output and path",
                "tamper": "off_chain canonical_payload is mutated for sampled rows before audit recompute",
                "payload_ablation": "proposed canonical_payload is reduced before commitment; baselines derived from same reduced payload",
                "context_tamper_fallback": "if input_summary is absent (e.g. output_only), context tampering falls back to output tampering",
                "model_tamper": "cascade: model_summary.inference_config, else governance_summary.policy_version, else input modality, else output_label",
                "governance_tamper": "cascade: governance_summary.policy_version, else input modality, else output_label",
            },
        }
        if str(tamper_strategy) == "layer_quarter":
            summary["notes"]["layer_quarter_tamper"] = (
                "non-cascade rotation by row index: model, governance, context, output; "
                "absent blocks => no-op => proposed tamper-detection often ~1.0 / ~0.75 / ~0.25 "
                "for full / no_model / output_only at large n."
            )
        if str(tamper_strategy) == "salt_corrupt":
            summary["notes"]["salt_corrupt"] = (
                "Auditor recomputation uses corrupted salt_bind (+|wrong_salt_bind) on tampered rows; "
                "canonical_payload unchanged; commitment used correct salt at dataset commit time."
            )
        if str(tamper_strategy) == "field_swap":
            summary["notes"]["field_swap"] = (
                "Swap model_summary <-> governance_summary to simulate cross-field binding confusion."
            )
        if str(tamper_strategy) == "truncate_partial":
            summary["notes"]["truncate_partial"] = (
                "Truncate output_label and long digests — silent shortening / truncation attack."
            )
        if str(tamper_strategy) == "cross_sample_splice":
            summary["notes"]["cross_sample_splice"] = (
                "Replace output_summary with another row's output_summary (deterministic peer index)."
            )
        (records_dir / "experiment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_baseline_commitment_specs(
            records_dir,
            pending[0]["payload_orig"],  # type: ignore[arg-type]
            str(pending[0]["pred_output"]),
            str(pending[0]["image_path"]),
        )

    performance_overhead = build_performance_overhead(
        records_dir,
        n,
        audit_total_ms,
        _perf_t0,
        _cpu_t0,
    )
    exp_summary_path = records_dir / "experiment_summary.json"
    if exp_summary_path.is_file():
        try:
            sd = json.loads(exp_summary_path.read_text(encoding="utf-8"))
            sd["performance_overhead"] = performance_overhead
            exp_summary_path.write_text(json.dumps(sd, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

    _csv_fp = _input_csv_stat_fingerprint(input_csv)
    assumptions = {
        "note": "gas_used is an estimate baseline, not on-chain measured receipt gasUsed.",
        "records_layout": records_layout,
        "pipeline_role": pipeline_role,
        "row_count_this_run": int(n),
        "max_rows_arg": int(max_rows),
        "input_csv_size_bytes": int(_csv_fp) if _csv_fp != "missing" else None,
        "payload_ablation": payload_ablation,
        "canonical_dataset_dir": str(dataset_dir_for_ref) if dataset_dir_for_ref else "",
        "shared_dataset_dir": meta_shared_dataset_dir,
        "gas_used_fixed": int(gas_used_fixed),
        "gas_used_range": [int(x) for x in gas_used_range] if gas_used_range else [int(gas_used_fixed)] * 3,
        "gas_price_gwei": float(gas_price_gwei),
        "eth_usd": float(eth_usd),
        "policy_version": str(policy_version),
        "prompt_template_version": str(prompt_template_version),
        "app_scene": str(app_scene),
        "domain_separator": str(domain_separator),
        "simulate_chain_delay": bool(simulate_chain_delay),
        "tamper_rate": float(tamper_rate),
        "tamper_strategy": str(tamper_strategy),
        "write_baseline_experiment": bool(write_baseline_experiment),
        "input_csv": str(input_csv),
        "random_seed": random_seed,
        "performance_overhead": performance_overhead,
    }
    (records_dir / "assumptions.json").write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8")

    if dataset_dir_for_ref is not None:
        _write_shared_ref(records_dir, dataset_dir_for_ref, "dataset")
        _try_symlink(dataset_dir_for_ref / "commitments_chain.csv", records_dir / "commitments_chain.csv")
        _try_symlink(dataset_dir_for_ref / "inference_offchain.jsonl", records_dir / "inference_offchain.jsonl")
