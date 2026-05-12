import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

from src.audit_pipeline import _dataset_key_from_inputs, run_pipeline_from_csv
from src.fetch_json_http import coingecko_demo_headers, fetch_etherscan_propose_gas_gwei, fetch_json

# -----------------------------------------------------------------------------
# Default --input-csv: intentionally a placeholder string, not a host path.
# Pass --input-csv /path/to/your.csv or set env CSV (or INPUT_CSV) before running.
# -----------------------------------------------------------------------------


def _default_input_csv() -> str:
    return (os.environ.get("CSV") or os.environ.get("INPUT_CSV") or "<PATH_TO_INFERENCE_RESULTS_CSV>").strip()


def _resolve_shared_dataset_dir(
    records_root: Path,
    dataset_key: str,
    shared_dir_arg: str,
    shared_tag_arg: str,
) -> Path | None:
    if shared_dir_arg.strip():
        p = Path(shared_dir_arg).expanduser().resolve()
        if (p / "commitments_chain.csv").is_file():
            return p
        return None
    if shared_tag_arg.strip():
        tag_dir = (records_root / shared_tag_arg.strip()).resolve()
        refp = tag_dir / "shared_dataset_ref.json"
        if refp.is_file():
            data = json.loads(refp.read_text(encoding="utf-8"))
            p = Path(str(data.get("shared_dataset_dir", ""))).resolve()
            if p.is_dir() and (p / "commitments_chain.csv").is_file():
                return p
        if (tag_dir / "commitments_chain.csv").is_file():
            return tag_dir
        # Tag ref missing or stale path (e.g. copied from another host): fall back to key dir.
    ds = (records_root / "_datasets" / dataset_key).resolve()
    if (ds / "commitments_chain.csv").is_file():
        return ds
    return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch_json(url: str, timeout_sec: int = 10) -> dict:
    with urlopen(url, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_live_eth_usd(default_value: float) -> tuple[float, str]:
    errors: list[str] = []
    # Source 1: CoinGecko simple price (optional COINGECKO_API_KEY / COINGECKO_DEMO_API_KEY for Demo tier)
    try:
        data = fetch_json(
            "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
            timeout_sec=10.0,
            headers=coingecko_demo_headers(),
        )
        return float(data["ethereum"]["usd"]), "live:coingecko"
    except Exception as exc:
        errors.append(f"coingecko:{exc}")

    # Source 2: Coinbase spot price
    try:
        data = _fetch_json("https://api.coinbase.com/v2/prices/ETH-USD/spot")
        return float(data["data"]["amount"]), "live:coinbase"
    except Exception as exc:
        errors.append(f"coinbase:{exc}")

    # Source 3: CryptoCompare
    try:
        data = _fetch_json("https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms=USD")
        return float(data["USD"]), "live:cryptocompare"
    except Exception as exc:
        errors.append(f"cryptocompare:{exc}")

    return float(default_value), "fallback:arg|" + ";".join(errors)


def _fetch_live_gas_price_gwei(default_value: float) -> tuple[float, str]:
    # 1) Etherscan gas oracle (needs ETHERSCAN_API_KEY — not CoinGecko).
    # 2) Public RPC eth_gasPrice (often 403 / needs provider key).
    import urllib.request

    errors: list[str] = []
    try:
        gwei, label = fetch_etherscan_propose_gas_gwei()
        return gwei, label
    except Exception as exc:
        errors.append(f"etherscan:{exc}")

    rpc_endpoints = [
        ("cloudflare-eth-rpc", "https://cloudflare-eth.com"),
        ("llama-eth-rpc", "https://eth.llamarpc.com"),
        ("ankr-eth-rpc", "https://rpc.ankr.com/eth"),
    ]
    for name, url in rpc_endpoints:
        try:
            req = urllib.request.Request(
                url,
                method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {"jsonrpc": "2.0", "method": "eth_gasPrice", "params": [], "id": 1}
                ).encode("utf-8"),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload:
                raise ValueError(str(payload["error"]))
            wei_hex = payload.get("result", "0x0")
            wei_val = int(wei_hex, 16)
            gwei = wei_val / 1e9
            if gwei <= 0:
                raise ValueError("non-positive gwei")
            return gwei, f"live:{name}"
        except Exception as exc:
            errors.append(f"{name}:{exc}")

    return float(default_value), "fallback:arg|" + ";".join(errors)


def resolve_gas_eth_for_pipeline(med_ns: argparse.Namespace, use_live: bool) -> tuple[float, float]:
    """Resolve gas_price_gwei and eth_usd from med defaults, optionally via live oracles."""
    gas_price_gwei = float(med_ns.gas_price_gwei)
    eth_usd = float(med_ns.eth_usd)
    if use_live:
        gas_price_gwei, _ = _fetch_live_gas_price_gwei(gas_price_gwei)
        eth_usd, _ = _fetch_live_eth_usd(eth_usd)
    return gas_price_gwei, eth_usd


def _parse_gas_range(raw: str) -> list[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--gas-used-range must have exactly 3 integers: low,mid,high")
    values = [int(x) for x in parts]
    return sorted(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run commitment audit using real medical inference CSV.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=_default_input_csv(),
        help="Path to inference result CSV (default: env CSV or INPUT_CSV, else literal placeholder—must be overridden).",
    )
    parser.add_argument("--tag", type=str, default="med_real_no_tamper", help="Output subfolder tag.")
    parser.add_argument("--tamper-rate", type=float, default=0.0, help="Tamper ratio in [0,1].")
    parser.add_argument(
        "--tamper-strategy",
        type=str,
        choices=[
            "output",
            "context",
            "mixed",
            "model",
            "governance",
            "layer_quarter",
            "salt_corrupt",
            "field_swap",
            "truncate_partial",
            "cross_sample_splice",
        ],
        default="output",
        help="How off-chain canonical_payload is mutated for tampered rows. "
        "layer_quarter = rotate model/gov/context/output by row (no cascade); useful for payload-ablation sweeps.",
    )
    parser.add_argument(
        "--no-baseline-experiment",
        action="store_true",
        help="Skip experiment_baselines.csv and experiment_summary.json.",
    )
    parser.add_argument(
        "--gas-used-fixed",
        type=int,
        default=62000,
        help="Estimated gasUsed baseline per record (not real receipt gasUsed).",
    )
    parser.add_argument(
        "--gas-used-range",
        type=str,
        default="42000,62000,82000",
        help="Estimated gasUsed range low,mid,high (comma separated).",
    )
    parser.add_argument("--gas-price-gwei", type=float, default=0.10, help="Gas price in gwei (manual fallback).")
    parser.add_argument("--eth-usd", type=float, default=3000.0, help="ETH/USD price (manual fallback).")
    parser.add_argument(
        "--use-live-market-data",
        action="store_true",
        help="Fetch live gasPrice and ETH/USD; fallback to CLI values on failure.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to process (0 means all rows).")
    parser.add_argument(
        "--simulate-chain-delay",
        action="store_true",
        help="If set, sleep per row to mimic chain write latency.",
    )
    parser.add_argument("--model-name", type=str, default="med_disease_cnn_router", help="Model identifier.")
    parser.add_argument(
        "--model-version",
        type=str,
        default="med_disease_cnn_router_v1",
        help="Version string used to derive model hash.",
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default="pred_from_csv,topk=3,source=multimodal_zh_5ep_routed",
        help="Inference config summary text.",
    )
    parser.add_argument("--policy-version", type=str, default="med_policy_v1", help="Governance policy version tag.")
    parser.add_argument(
        "--prompt-template-version",
        type=str,
        default="prompt_tpl_v1",
        help="Prompt template version used by medical model interface.",
    )
    parser.add_argument("--app-scene", type=str, default="medical_diagnosis", help="Application scene label.")
    parser.add_argument(
        "--domain-separator",
        type=str,
        default="MED-AUDIT-V1",
        help="Domain separator appended in commitment construction.",
    )
    parser.add_argument(
        "--records-layout",
        type=str,
        choices=["flat", "split"],
        default="split",
        help="split => canonical chain+jsonl under records/_datasets/<key>; tamper/cost tags reuse it.",
    )
    parser.add_argument(
        "--shared-dataset-dir",
        type=str,
        default="",
        help="Explicit canonical dataset directory (contains commitments_chain.csv). Overrides auto lookup.",
    )
    parser.add_argument(
        "--shared-dataset-tag",
        type=str,
        default="",
        help="Optional tag folder: follow its shared_dataset_ref.json, or use that folder if flat artifacts live there.",
    )
    parser.add_argument(
        "--payload-ablation",
        type=str,
        choices=["full", "no_governance", "no_model", "output_only"],
        default="full",
        help="Ablate proposed canonical_payload before commitment (split canonical key includes this).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent
    output_root = project_dir / "outputs"
    records_dir = output_root / "records" / args.tag
    records_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.is_file():
        raise SystemExit(
            f"Input CSV not found: {input_csv}\n"
            "Pass a real path with --input-csv, or set environment variable CSV (or INPUT_CSV). "
            "The default value in this repo is a placeholder only; see README."
        )
    model_weights_sha256 = _sha256(args.model_version)
    gas_used_range = _parse_gas_range(args.gas_used_range)
    gas_used_fixed = gas_used_range[1]
    gas_price_gwei = float(args.gas_price_gwei)
    eth_usd = float(args.eth_usd)
    gas_source = "arg"
    eth_source = "arg"

    if args.use_live_market_data:
        gas_price_gwei, gas_source = _fetch_live_gas_price_gwei(gas_price_gwei)
        eth_usd, eth_source = _fetch_live_eth_usd(eth_usd)

    records_root = output_root / "records"
    records_layout = str(args.records_layout).lower().strip()
    dataset_key = _dataset_key_from_inputs(
        input_csv,
        int(args.max_rows),
        args.model_name,
        model_weights_sha256,
        args.inference_config,
        args.policy_version,
        args.prompt_template_version,
        args.app_scene,
        args.domain_separator,
        str(args.payload_ablation),
    )
    tag_lower = str(args.tag).lower()
    pipeline_role = "full"
    shared_resolved: Path | None = None
    if records_layout == "split":
        if args.tamper_rate > 0:
            pipeline_role = "tamper_from_shared"
            shared_resolved = _resolve_shared_dataset_dir(
                records_root,
                dataset_key,
                str(args.shared_dataset_dir),
                str(args.shared_dataset_tag),
            )
            if shared_resolved is None:
                raise SystemExit(
                    "split layout + tamper requires an existing canonical dataset (same CSV, --max-rows, model, "
                    "policy, prompt, app_scene, domain-separator as the canonical run). "
                    "Run a no-tamper tag first, or pass --shared-dataset-dir / --shared-dataset-tag. "
                    f"Default lookup: {records_root / '_datasets' / dataset_key}"
                )
        elif ("cost" in tag_lower) or ("live" in tag_lower):
            pipeline_role = "cost_from_shared"
            shared_resolved = _resolve_shared_dataset_dir(
                records_root,
                dataset_key,
                str(args.shared_dataset_dir),
                str(args.shared_dataset_tag),
            )
            if shared_resolved is None:
                raise SystemExit(
                    "split layout + cost/live tag requires a canonical dataset to reprice. "
                    "Run a no-tamper tag first, or pass --shared-dataset-dir / --shared-dataset-tag. "
                    f"Default lookup: {records_root / '_datasets' / dataset_key}"
                )
        else:
            pipeline_role = "full"
    else:
        records_layout = "flat"

    run_pipeline_from_csv(
        input_csv=input_csv,
        records_dir=records_dir,
        model_name=args.model_name,
        model_weights_sha256=model_weights_sha256,
        inference_config=args.inference_config,
        tamper_rate=args.tamper_rate,
        max_rows=args.max_rows,
        simulate_chain_delay=args.simulate_chain_delay,
        gas_used_fixed=gas_used_fixed,
        gas_used_range=gas_used_range,
        gas_price_gwei=gas_price_gwei,
        eth_usd=eth_usd,
        policy_version=args.policy_version,
        prompt_template_version=args.prompt_template_version,
        app_scene=args.app_scene,
        domain_separator=args.domain_separator,
        tamper_strategy=str(args.tamper_strategy),
        write_baseline_experiment=not bool(args.no_baseline_experiment),
        records_layout=records_layout,
        pipeline_role=pipeline_role,
        shared_dataset_dir=shared_resolved,
        payload_ablation=str(args.payload_ablation),
        random_seed=None,
    )

    if args.tamper_rate > 0 and str(args.tamper_strategy) == "context":
        run_profile = "ctx_tamper_eval"
        run_purpose = "Context-only tamper scenario: highlight output-only baseline blind spot."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "model":
        run_profile = "model_meta_tamper_eval"
        run_purpose = (
            "Binding-stress tamper with --tamper-strategy model: cascade model -> governance -> context -> output "
            "so ablations do not all reduce to the same output-only attack."
        )
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "governance":
        run_profile = "gov_meta_tamper_eval"
        run_purpose = "Governance-metadata tamper: mutate policy_version (ablation-sensitive vs no_governance)."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "layer_quarter":
        run_profile = "layer_quarter_ablation_eval"
        run_purpose = (
            "Quarter-layer tamper (no cascade): row i uses layer i%%4 so ablations drop absent blocks; "
            "proposed detection typically steps ~100%% / ~75%% / ~25%% for full / no_model / output_only."
        )
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "salt_corrupt":
        run_profile = "salt_bind_corrupt_eval"
        run_purpose = "Salt-binding attack: auditor recomputes with corrupted salt_bind on tampered rows."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "field_swap":
        run_profile = "field_swap_eval"
        run_purpose = "Swap model_summary vs governance_summary across tampered rows."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "truncate_partial":
        run_profile = "truncate_partial_eval"
        run_purpose = "Partial truncation of output_label / digests on tampered rows."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "cross_sample_splice":
        run_profile = "cross_sample_splice_eval"
        run_purpose = "Cross-sample splice: paste another row's output_summary onto tampered rows."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "mixed":
        run_profile = "mixed_tamper_eval"
        run_purpose = "Mixed tamper: alternate output vs context mutations across rows."
    elif args.tamper_rate > 0 and str(args.tamper_strategy) == "output":
        run_profile = "out_tamper_eval"
        run_purpose = "Output tamper scenario: verify detection under direct output manipulation."
    elif pipeline_role == "cost_from_shared":
        run_profile = "cost_eval"
        run_purpose = "Cost sensitivity scenario: low/mid/high gas and market-driven fee analysis."
    else:
        run_profile = "no_tamper_baseline"
        run_purpose = "No-tamper integrity baseline scenario."

    market_meta = {
        "use_live_market_data": bool(args.use_live_market_data),
        "gas_used_range": gas_used_range,
        "gas_used_mid_for_fee_est_eth": gas_used_fixed,
        "gas_price_gwei": gas_price_gwei,
        "gas_price_source": gas_source,
        "eth_usd": eth_usd,
        "eth_usd_source": eth_source,
        "policy_version": args.policy_version,
        "prompt_template_version": args.prompt_template_version,
        "app_scene": args.app_scene,
        "domain_separator": args.domain_separator,
        "run_profile": run_profile,
        "run_purpose": run_purpose,
    }
    (records_dir / "market_data.json").write_text(json.dumps(market_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rec_primary = [
        "shared_dataset_ref.json",
        "tamper_audit.csv",
        "experiment_baselines.csv",
        "experiment_summary.json",
        "assumptions.json",
        "market_data.json",
        "baseline_commitment_specs.json",
    ]
    if str(args.records_layout).lower() == "split" and pipeline_role == "full":
        rec_primary = [
            f"../_datasets/{dataset_key}/commitments_chain.csv",
            f"../_datasets/{dataset_key}/inference_offchain.jsonl",
        ] + rec_primary
    elif str(args.records_layout).lower() == "split" and pipeline_role == "tamper_from_shared":
        rec_primary = [
            "commitments_chain.csv (symlink to canonical)",
            "inference_offchain.jsonl (tampered, local)",
        ] + rec_primary
    elif str(args.records_layout).lower() == "split" and pipeline_role == "cost_from_shared":
        rec_primary = [
            "commitments_chain.csv (repriced, local)",
            "inference_offchain.jsonl (symlink to canonical)",
        ] + rec_primary
    else:
        rec_primary = [
            "commitments_chain.csv",
            "inference_offchain.jsonl",
        ] + rec_primary

    run_guide = {
        "tag": args.tag,
        "run_profile": run_profile,
        "run_purpose": run_purpose,
        "records_dir": str(records_dir),
        "records_layout": str(args.records_layout),
        "pipeline_role": pipeline_role,
        "payload_ablation": str(args.payload_ablation),
        "dataset_key": dataset_key,
        "records_primary_files": rec_primary,
    }
    (records_dir / "run_guide.json").write_text(json.dumps(run_guide, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Medical CSV integration done.")
    print(f"Input CSV: {input_csv}")
    print(f"Records: {records_dir}")
    print(f"gas_price_gwei={gas_price_gwei} ({gas_source}), eth_usd={eth_usd} ({eth_source})")


if __name__ == "__main__":
    main()
