"""Helius API client — transaction fetching, balances, caching."""

import json
import os
import time
from typing import Dict, List, Optional

import requests

from config import (
    HELIUS_API_KEY,
    HELIUS_MAX_RETRIES,
    HELIUS_MIN_DELAY_SEC,
    HELIUS_PAGE_LIMIT,
    SOL_MINT,
    STABLE_MINTS,
)
from models import ensure_dir, sf


# =========================
# Token amount parsing
# =========================

def parse_token_amount(raw) -> Optional[float]:
    if not isinstance(raw, dict):
        return None
    inner = raw.get("rawTokenAmount") or raw
    if isinstance(inner, dict):
        ui = inner.get("uiAmount")
        if ui is not None:
            try:
                return float(ui)
            except Exception:
                pass
    ta = raw.get("tokenAmount")
    dec = raw.get("decimals")
    if ta is None:
        return None
    try:
        return int(str(ta)) / (10 ** int(dec))
    except Exception:
        return None


# =========================
# HTTP helpers
# =========================

def _h_get(url: str, session: requests.Session):
    for i in range(HELIUS_MAX_RETRIES):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
                continue
            # 400/404 = permanent error (bad address, etc.) — don't retry
            if r.status_code in (400, 404):
                raise RuntimeError(
                    f"Helius {r.status_code}: {r.text[:200]}"
                )
            r.raise_for_status()
            return r.json()
        except RuntimeError:
            raise  # re-raise permanent errors immediately
        except Exception as e:
            print(f"  Helius err ({i + 1}): {e}")
            time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
    raise RuntimeError("Helius failed after retries")


def _h_post(url: str, payload: dict, session: requests.Session):
    for i in range(HELIUS_MAX_RETRIES):
        try:
            r = session.post(url, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
                continue
            if r.status_code in (400, 404):
                raise RuntimeError(
                    f"Helius {r.status_code}: {r.text[:200]}"
                )
            r.raise_for_status()
            return r.json()
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  Helius err ({i + 1}): {e}")
            time.sleep(HELIUS_MIN_DELAY_SEC * (2 ** i))
    raise RuntimeError("Helius failed after retries")


# =========================
# HeliusClient
# =========================

class HeliusClient:
    def __init__(self, api_key: str, cache_dir: str, use_cache: bool):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.session = requests.Session()

    @property
    def _rpc_url(self) -> str:
        return f"https://mainnet.helius-rpc.com/?api-key={self.api_key}"

    def fetch_transactions(self, wallet: str, end_ts: int) -> List[dict]:
        cd = os.path.join(self.cache_dir, wallet[:8])
        ensure_dir(cd)
        before = None
        all_txs = []
        page = 0
        while True:
            page += 1
            base = (
                f"https://api-mainnet.helius-rpc.com/v0/addresses/{wallet}/transactions"
                f"?api-key={self.api_key}&limit={HELIUS_PAGE_LIMIT}"
            )
            url = base + (f"&before={before}" if before else "")
            cp = os.path.join(cd, f"page_{page:04d}.json")
            if self.use_cache and os.path.exists(cp):
                with open(cp) as f:
                    txs = json.load(f)
            else:
                txs = _h_get(url, self.session)
                if self.use_cache:
                    with open(cp, "w") as f:
                        json.dump(txs, f)
                time.sleep(HELIUS_MIN_DELAY_SEC)
            if not txs:
                break
            for tx in txs:
                ts = tx.get("timestamp")
                if isinstance(ts, int) and ts <= end_ts:
                    all_txs.append(tx)
            before = txs[-1].get("signature")
            if not before:
                break
            if page % 10 == 0:
                print(f"    p{page}: {len(all_txs)} txs")
        return all_txs

    def fetch_balances(self, wallet: str) -> Dict[str, float]:
        bals = {}
        try:
            r = _h_post(
                self._rpc_url,
                {"jsonrpc": "2.0", "id": "s", "method": "getBalance",
                 "params": [wallet]},
                self.session,
            )
            sol = r.get("result", {}).get("value", 0) / 1e9
            if sol > 0:
                bals[SOL_MINT] = sol

            r = _h_post(
                self._rpc_url,
                {"jsonrpc": "2.0", "id": "t", "method": "getTokenAccountsByOwner",
                 "params": [
                     wallet,
                     {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                     {"encoding": "jsonParsed"},
                 ]},
                self.session,
            )
            for a in r.get("result", {}).get("value", []):
                info = (a.get("account", {}).get("data", {})
                        .get("parsed", {}).get("info", {}))
                m = info.get("mint")
                ui = sf(info.get("tokenAmount", {}).get("uiAmount"), 0)
                if m and ui > 1e-6:
                    bals[m] = ui
            return bals
        except Exception as e:
            print(f"  Bal error: {e}")
            return {}

    def get_asset_price(self, mint: str) -> float:
        try:
            payload = {
                "jsonrpc": "2.0", "id": "p", "method": "getAsset",
                "params": {"id": mint},
            }
            r = _h_post(self._rpc_url, payload, self.session)
            if "result" in r:
                return sf(
                    r["result"]
                    .get("token_info", {})
                    .get("price_info", {})
                    .get("price_per_token"),
                    0,
                )
        except Exception:
            pass
        return 0.0
