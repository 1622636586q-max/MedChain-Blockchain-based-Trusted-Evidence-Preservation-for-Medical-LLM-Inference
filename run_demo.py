import argparse
from pathlib import Path

from src.audit_pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mock commitment pipeline (records only).")
    parser.add_argument("--num-requests", type=int, default=120, help="Number of mock requests.")
    parser.add_argument(
        "--tamper-rate",
        type=float,
        default=0.08,
        help="Tamper ratio in [0,1]. Use 0 for clean run.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Output subfolder tag, e.g. no_tamper or tamper_10.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent
    output_root = project_dir / "outputs"
    records_dir = output_root / "records" / args.tag
    records_dir.mkdir(parents=True, exist_ok=True)

    print("Running mock commitment pipeline...")
    run_pipeline(records_dir=records_dir, num_requests=args.num_requests, tamper_rate=args.tamper_rate)

    print("Done (records only).")
    print(f"Records: {records_dir}")


if __name__ == "__main__":
    main()
