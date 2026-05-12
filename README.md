# MedChain — Blockchain-based Trusted Evidence Preservation for Medical LLM Inference


Minimal repository: **mock ledger + commitment / audit pipeline + optional Ethereum deploy/submit**.  
There is **no** plotting stack (no matplotlib, no figure scripts, no PNG outputs from Python).

## Layout

| Path | Role |
|------|------|
| `src/audit_pipeline.py` | Canonical payload, commitments, tamper audit, mock chain CSV |
| `src/offchain_vault.py` | Optional encrypted off-chain artifacts |
| `src/fetch_json_http.py` | Etherscan / CoinGecko helpers for live gas / ETH-USD |
| `contracts/CommitmentRegistry.sol` | Solidity registry (real deployment) |
| `deploy_commitment_registry.py` | Compile + deploy (`web3`, `py-solc-x`) |
| `submit_commitments_onchain.py` | Submit rows from `commitments_chain.csv` |
| `run_med_csv_integration.py` | Medical CSV → pipeline + optional market data |
| `run_demo.py` / `run_paper_scenarios.py` | Small mock demos |
| `run_attack_detection_matrix.py` | Batch tamper strategies → **CSV + JSON** under `outputs/attack_matrix/` |
| `docs/` | English evaluation traceability tables (`evaluation_plan_scenarios.csv`, `evaluation_6_3_audit_scenarios.csv`) and `REMOTE_INPUT_CSV.md` |
| `vault_on.ps1` / `vault_off.ps1` | Toggle vault for the current PowerShell session |

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

Medical CSV (required: real path — the default `--input-csv` in code is a placeholder):

```bash
export CSV="/absolute/path/to/your_inference_results.csv"
python run_med_csv_integration.py --tag med_real_no_tamper --tamper-rate 0
# or: python run_med_csv_integration.py --input-csv "/absolute/path/to/your_inference_results.csv" ...
```

Attack matrix (after a canonical no-tamper tag exists):

```bash
python run_attack_detection_matrix.py --input-csv YOUR.csv --canonical-tag YOUR_NT_TAG
```

Vault (PowerShell):

```powershell
.\vault_on.ps1
.\vault_off.ps1
```

## Dependencies

`web3`, `py-solc-x`, `cryptography`.

**Optional live oracles** (`run_med_csv_integration.py --use-live-market-data`): set `ETHERSCAN_API_KEY` and optionally `COINGECKO_API_KEY` / `COINGECKO_DEMO_API_KEY`. Defaults in `src/fetch_json_http.py` are empty placeholders.
