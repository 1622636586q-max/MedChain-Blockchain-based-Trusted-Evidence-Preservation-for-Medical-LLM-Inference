from pathlib import Path

from src.audit_pipeline import run_pipeline


def _run_one(project_dir: Path, tag: str, tamper_rate: float, num_requests: int = 120) -> None:
    output_root = project_dir / "outputs"
    records_dir = output_root / "records" / tag
    records_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(records_dir=records_dir, num_requests=num_requests, tamper_rate=tamper_rate)

    print(f"[{tag}] tamper_rate={tamper_rate}")
    print(f"records: {records_dir}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    _run_one(project_dir, tag="no_tamper", tamper_rate=0.0)
    _run_one(project_dir, tag="tamper_10pct", tamper_rate=0.10)


if __name__ == "__main__":
    main()
