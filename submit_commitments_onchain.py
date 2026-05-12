import argparse
import csv
import hashlib
import json
from pathlib import Path

from web3 import Web3


def _bytes32_from_hex(hex_str: str) -> bytes:
    s = (hex_str or "").strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) != 64:
        raise ValueError(f"Expected 32-byte hex (64 chars), got len={len(s)}")
    return bytes.fromhex(s)


def _request_id_hash_hex(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit commitments CSV rows to deployed contract.")
    parser.add_argument("--rpc-url", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--chain-id", type=int, required=True)
    parser.add_argument("--contract-address", default="", help="Optional; auto-read from deployment json if omitted.")
    parser.add_argument(
        "--contract-abi-json",
        default="outputs/onchain/registry_deploy.json",
        help="Path to deployment json with `abi` and optionally `contract_address`.",
    )
    parser.add_argument("--input-csv", required=True, help="Path to commitments_chain.csv.")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional output CSV path with on-chain receipts. Default: <input>_onchain.csv",
    )
    args = parser.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"RPC not reachable: {args.rpc_url}")

    abi_json_path = Path(args.contract_abi_json).resolve()
    abi_payload = json.loads(abi_json_path.read_text(encoding="utf-8"))
    abi = abi_payload["abi"]
    contract_address = (args.contract_address or "").strip() or str(abi_payload.get("contract_address", "")).strip()
    if not contract_address:
        raise ValueError("contract address not provided and not found in deployment json")
    contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
    acct = w3.eth.account.from_key(args.private_key)

    in_csv = Path(args.input_csv).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else in_csv.with_name(in_csv.stem + "_onchain.csv")

    rows_out = []
    nonce = w3.eth.get_transaction_count(acct.address)
    gas_price = w3.eth.gas_price

    with in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if args.max_rows > 0 and idx >= args.max_rows:
                break

            request_id = row["request_id"]
            commitment_hash = row["commitment_hash"]
            model_hash = row["model_weights_sha256"]
            req_hash = _request_id_hash_hex(request_id)

            tx = contract.functions.recordCommitment(
                _bytes32_from_hex(req_hash),
                _bytes32_from_hex(commitment_hash),
                _bytes32_from_hex(model_hash),
            ).build_transaction(
                {
                    "from": acct.address,
                    "nonce": nonce,
                    "chainId": args.chain_id,
                    "gasPrice": gas_price,
                }
            )
            tx["gas"] = w3.eth.estimate_gas(tx)
            signed = w3.eth.account.sign_transaction(tx, args.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

            row_out = dict(row)
            row_out["onchain_request_id_hash"] = req_hash
            row_out["onchain_tx_hash"] = receipt.transactionHash.hex()
            row_out["onchain_status"] = int(receipt.status)
            row_out["onchain_block_number"] = int(receipt.blockNumber)
            row_out["onchain_gas_used"] = int(receipt.gasUsed)
            rows_out.append(row_out)
            nonce += 1

            if (idx + 1) % 200 == 0:
                print(f"Submitted {idx+1} rows...")

    if not rows_out:
        raise RuntimeError("No rows submitted.")

    fieldnames = list(rows_out[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("On-chain submission completed.")
    print(f"Input: {in_csv}")
    print(f"Output: {out_csv}")
    print(f"Rows: {len(rows_out)}")
    print(f"Contract: {contract_address}")
    print(f"ABI JSON: {abi_json_path}")


if __name__ == "__main__":
    main()
