"""HTTP GET helpers for CoinGecko (ETH/USD) and Etherscan (gas oracle).

API keys must come from environment variables for public clones — embedded
defaults are empty placeholders only. Set COINGECKO_API_KEY or
COINGECKO_DEMO_API_KEY, and ETHERSCAN_API_KEY, before relying on live oracles.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen

# Placeholder defaults only (never put real keys in a public repo).
_DEFAULT_COINGECKO_DEMO_KEY = ""
_DEFAULT_ETHERSCAN_API_KEY = ""


def coingecko_demo_headers() -> dict[str, str]:
    """Sets x-cg-demo-api-key if env or embedded default is non-empty."""
    key = (
        os.environ.get("COINGECKO_API_KEY")
        or os.environ.get("COINGECKO_DEMO_API_KEY")
        or _DEFAULT_COINGECKO_DEMO_KEY
    ).strip()
    if not key:
        return {}
    return {"x-cg-demo-api-key": key}


def etherscan_api_key() -> str:
    return (os.environ.get("ETHERSCAN_API_KEY") or _DEFAULT_ETHERSCAN_API_KEY).strip()


def fetch_json(url: str, *, timeout_sec: float = 12.0, headers: dict[str, str] | None = None) -> dict[str, Any]:
    merged: dict[str, str] = dict(headers or {})
    req = Request(url, headers=merged)
    with urlopen(req, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _etherscan_gasoracle_v2_url(api_key: str) -> str:
    """V1 api.etherscan.io/api was deprecated Aug 2025; V2 requires chainid."""
    return (
        "https://api.etherscan.io/v2/api"
        f"?chainid=1&module=gastracker&action=gasoracle&apikey={api_key}"
    )


def fetch_etherscan_propose_gas_gwei() -> tuple[float, str]:
    """Mainnet suggested gas from Etherscan Gas Tracker (ProposeGasPrice in gwei)."""
    key = etherscan_api_key()
    if not key:
        raise ValueError("ETHERSCAN_API_KEY missing (CoinGecko key does not supply Ethereum gas).")
    url = _etherscan_gasoracle_v2_url(key)
    data = fetch_json(url, timeout_sec=18.0)
    if str(data.get("status")) != "1":
        res = data.get("result")
        msg = data.get("message", "")
        raise ValueError(f"Etherscan gasoracle: message={msg!r} result={res!r}")
    result = data.get("result")
    if isinstance(result, str):
        raise ValueError(result)
    if not isinstance(result, dict):
        raise ValueError("unexpected gasoracle result shape")
    raw = result.get("ProposeGasPrice", "0")
    propose = float(str(raw).replace(",", ""))
    if propose <= 0:
        raise ValueError("invalid ProposeGasPrice")
    return propose, "live:etherscan-gasoracle"
