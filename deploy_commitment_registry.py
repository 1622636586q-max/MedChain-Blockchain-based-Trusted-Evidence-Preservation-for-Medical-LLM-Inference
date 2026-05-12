import argparse
import json
from pathlib import Path

from solcx import compile_standard, install_solc
from web3 import Web3


def compile_contract(contract_path: Path, solc_version: str) -> tuple[list[dict], str]:
    install_solc(solc_version)
    source = contract_path.read_text(encoding="utf-8")
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {contract_path.name: {"content": source}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode.object"],
                    }
                }
            },
        },
        solc_version=solc_version,
    )
    obj = compiled["contracts"][contract_path.name]["CommitmentRegistry"]
    abi = obj["abi"]
    bytecode = obj["evm"]["bytecode"]["object"]
    return abi, bytecode


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy CommitmentRegistry to EVM chain.")
    parser.add_argument("--rpc-url", required=True, help="EVM RPC endpoint.")
    parser.add_argument("--private-key", required=True, help="Deployer private key.")
    parser.add_argument("--chain-id", type=int, required=True, help="Chain ID (e.g., 31337).")
    parser.add_argument("--solc-version", default="0.8.20", help="Solc version.")
    parser.add_argument(
        "--out-json",
        default="outputs/onchain/registry_deploy.json",
        help="Output deployment metadata json path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    contract_path = root / "contracts" / "CommitmentRegistry.sol"
    abi, bytecode = compile_contract(contract_path, args.solc_version)

    w3 = Web3(Web3.HTTPProvider(args.rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"RPC not reachable: {args.rpc_url}")

    acct = w3.eth.account.from_key(args.private_key)
    nonce = w3.eth.get_transaction_count(acct.address)
    gas_price = w3.eth.gas_price

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor().build_transaction(
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
    if receipt.status != 1:
        raise RuntimeError("Deployment failed")

    out = {
        "rpc_url": args.rpc_url,
        "chain_id": args.chain_id,
        "deployer": acct.address,
        "tx_hash": receipt.transactionHash.hex(),
        "contract_address": receipt.contractAddress,
        "gas_used": int(receipt.gasUsed),
        "abi": abi,
    }
    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Deployment succeeded.")
    print(f"Contract: {receipt.contractAddress}")
    print(f"Tx: {receipt.transactionHash.hex()}")
    print(f"Out: {out_path}")


if __name__ == "__main__":
    main()
