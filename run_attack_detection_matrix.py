"""
Run multiple tamper strategies against one canonical (split-layout) dataset and
emit a detection-rate matrix (CSV + JSON).

Requires an existing no-tamper tag whose shared_dataset_ref points to the same
dataset key as --payload-ablation / CSV / max-rows settings.

Example:
  python run_attack_detection_matrix.py \\
    --input-csv <PATH_TO_INFERENCE_RESULTS_CSV> \\
    --canonical-tag YOUR_NO_TAMPER_TAG \\
    --tamper-rate 0.1 \\
    --matrix-basename attack_detection_matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from run_med_csv_integration import (
    _build_parser as med_build_parser,
    _parse_gas_range,
    _resolve_shared_dataset_dir,
    _sha256,
    resolve_gas_eth_for_pipeline,
)
from src.audit_pipeline import _dataset_key_from_inputs, run_pipeline_from_csv


DEFAULT_ATTACKS = [
    "output",
    "context",
    "salt_corrupt",
    "field_swap",
    "truncate_partial",
    "cross_sample_splice",
    "layer_quarter",
    "model",
]


def _pick_float(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return default


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch tamper strategies -> detection matrix.")
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--canonical-tag", type=str, required=True, help="Existing *_nt tag with canonical dataset.")
    p.add_argument("--tamper-rate", type=float, default=0.1)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--payload-ablation", type=str, default="full", choices=["full", "no_governance", "no_model", "output_only"])
    p.add_argument(
        "--attacks",
        type=str,
        default=",".join(DEFAULT_ATTACKS),
        help="Comma-separated tamper_strategy names (must exist in run_med_csv_integration).",
    )
    p.add_argument("--tag-prefix", type=str, default="atk_mtx", help="Prefix for generated record tags.")
    p.add_argument("--out-dir", type=str, default="outputs/attack_matrix")
    p.add_argument("--matrix-basename", type=str, default="attack_detection_matrix")
    p.add_argument("--skip-run", action="store_true", help="Only aggregate existing experiment_summary.json paths (advanced).")
    p.add_argument("--tag-suffix", type=str, default="", help="Optional suffix appended to each generated record tag.")
    p.add_argument(
        "--fixed-cost-rates",
        action="store_true",
        help="Use med CLI --gas-price-gwei / --eth-usd only (no HTTP). Default: live Etherscan/CoinGecko/oracles.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent

    attacks = [x.strip() for x in str(args.attacks).split(",") if x.strip()]
    if not attacks:
        raise SystemExit("No attacks listed.")

    input_csv = Path(args.input_csv).resolve()
    records_root = project_dir / "outputs" / "records"
    out_dir = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    med_ns = med_build_parser().parse_args([])
    model_weights_sha256 = _sha256(str(med_ns.model_version))
    dataset_key = _dataset_key_from_inputs(
        input_csv,
        int(args.max_rows),
        med_ns.model_name,
        model_weights_sha256,
        med_ns.inference_config,
        med_ns.policy_version,
        med_ns.prompt_template_version,
        med_ns.app_scene,
        med_ns.domain_separator,
        str(args.payload_ablation),
    )

    use_live = not bool(args.fixed_cost_rates)
    gas_price_gwei, eth_usd = resolve_gas_eth_for_pipeline(med_ns, use_live)
    if use_live:
        print(f"[market] gas_price_gwei={gas_price_gwei:.6g} eth_usd={eth_usd:.4g} (live)")

    rows_out: list[dict[str, object]] = []
    tag_suffix = str(args.tag_suffix).strip()

    if not args.skip_run:
        shared_resolved = _resolve_shared_dataset_dir(
            records_root,
            dataset_key,
            "",
            str(args.canonical_tag),
        )
        if shared_resolved is None:
            raise SystemExit(
                f"Cannot resolve canonical dataset for key={dataset_key}. "
                f"Ensure tag {args.canonical_tag!r} exists and matches CSV/max-rows/ablation."
            )

        gas_used_range = _parse_gas_range(med_ns.gas_used_range)
        gas_used_fixed = gas_used_range[1]

        for atk in attacks:
            tag = f"{args.tag_prefix}_{atk}_t{int(round(float(args.tamper_rate) * 100))}{tag_suffix}"
            records_dir = project_dir / "outputs" / "records" / tag
            records_dir.mkdir(parents=True, exist_ok=True)
            run_pipeline_from_csv(
                input_csv=input_csv,
                records_dir=records_dir,
                model_name=med_ns.model_name,
                model_weights_sha256=model_weights_sha256,
                inference_config=med_ns.inference_config,
                tamper_rate=float(args.tamper_rate),
                max_rows=int(args.max_rows),
                simulate_chain_delay=False,
                gas_used_fixed=gas_used_fixed,
                gas_used_range=gas_used_range,
                gas_price_gwei=gas_price_gwei,
                eth_usd=eth_usd,
                policy_version=med_ns.policy_version,
                prompt_template_version=med_ns.prompt_template_version,
                app_scene=med_ns.app_scene,
                domain_separator=med_ns.domain_separator,
                tamper_strategy=str(atk),
                write_baseline_experiment=True,
                records_layout="split",
                pipeline_role="tamper_from_shared",
                shared_dataset_dir=shared_resolved,
                payload_ablation=str(args.payload_ablation),
                random_seed=None,
            )

            summary_path = records_dir / "experiment_summary.json"
            if not summary_path.is_file():
                raise SystemExit(f"Missing {summary_path}")
            d = json.loads(summary_path.read_text(encoding="utf-8"))
            ba = d.get("baseline_a") or {}
            bb = d.get("baseline_b") or {}
            rows_out.append(
                {
                    "attack": atk,
                    "tamper_rate": d.get("tamper_rate"),
                    "n_records": d.get("n_records"),
                    "tampered_count": d.get("tampered_count"),
                    "det_proposed": _pick_float(d, "tamper_detection_rate_proposed"),
                    "det_baseline_a": _pick_float(ba, "tamper_detection_rate"),
                    "det_baseline_b": _pick_float(bb, "tamper_detection_rate"),
                    "repro_pass_proposed": _pick_float(d, "reproducibility_pass_rate_proposed"),
                    "summary_path": str(summary_path.as_posix()),
                }
            )
    else:
        for atk in attacks:
            tag = f"{args.tag_prefix}_{atk}_t{int(round(float(args.tamper_rate) * 100))}{tag_suffix}"
            summary_path = project_dir / "outputs" / "records" / tag / "experiment_summary.json"
            if not summary_path.is_file():
                print(f"[warn] skip missing {summary_path}", file=sys.stderr)
                continue
            d = json.loads(summary_path.read_text(encoding="utf-8"))
            ba = d.get("baseline_a") or {}
            bb = d.get("baseline_b") or {}
            rows_out.append(
                {
                    "attack": atk,
                    "tamper_rate": d.get("tamper_rate"),
                    "n_records": d.get("n_records"),
                    "tampered_count": d.get("tampered_count"),
                    "det_proposed": _pick_float(d, "tamper_detection_rate_proposed"),
                    "det_baseline_a": _pick_float(ba, "tamper_detection_rate"),
                    "det_baseline_b": _pick_float(bb, "tamper_detection_rate"),
                    "repro_pass_proposed": _pick_float(d, "reproducibility_pass_rate_proposed"),
                    "summary_path": str(summary_path.as_posix()),
                }
            )

    csv_path = out_dir / f"{args.matrix_basename}.csv"
    json_path = out_dir / f"{args.matrix_basename}.json"
    fieldnames = [
        "attack",
        "tamper_rate",
        "n_records",
        "tampered_count",
        "det_proposed",
        "det_baseline_a",
        "det_baseline_b",
        "repro_pass_proposed",
        "summary_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    payload = {
        "input_csv": str(input_csv.as_posix()),
        "canonical_tag": str(args.canonical_tag),
        "dataset_key": dataset_key,
        "payload_ablation": str(args.payload_ablation),
        "tamper_rate": float(args.tamper_rate),
        "rows": rows_out,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
