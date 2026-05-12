# Remote host: default `--input-csv` for this MedChain repo

**Replace every angle-bracket placeholder with paths on your machine.**  
These examples are not tied to a fixed username or disk layout.

Use the same CSV path for `run_med_csv_integration.py`, `run_attack_detection_matrix.py`,
and any batch runs **unless you override `CSV` / `--input-csv`.**

```bash
export CSV="<REMOTE_PROJECT_ROOT>/<DATASET_DIR>/test_results_multimodal_zh_5ep_routed.csv"
```

**Note:** `experiments/med_vlm_auxiliary/outputs/med_disease_infer_test.csv` is a
different, small router or smoke table — do **not** substitute it for full-scale
runs unless you intend that subset.

**Project root on the same host (optional helper env):**

```bash
export BLOCKCHAIN_MEDCHAIN="<REMOTE_PROJECT_ROOT>/MedChain Blockchain-based Trusted Evidence Preservation for Medical LLM Inference"
```

**Note:** row count follows the CSV. `--max-rows 0` means “all rows in that file”.
If `experiment_summary.json` shows a small `n_records`, check row count on the
server for the file above.

**Same path, bigger file:** `dataset_key` includes the CSV file’s **byte size**. If you
replace the file in place with more rows, the key changes and split-layout runs
will use a **new** `outputs/records/_datasets/<key>/` after you re-run the `*_nt`
canonical steps.

**Still seeing a tiny `n_records`?**

1. **Confirm the file on disk** (path matches your `CSV`):

   `python -c "import csv; p='<REMOTE_PROJECT_ROOT>/<DATASET_DIR>/test_results_multimodal_zh_5ep_routed.csv'; print(sum(1 for _ in csv.DictReader(open(p,encoding='utf-8'))))"`

2. **Stderr:** each run prints  
   `[audit_pipeline] row_count_this_run=... max_rows=...`.
3. **`assumptions.json`:** `row_count_this_run`, `max_rows_arg`, `input_csv_size_bytes`.
