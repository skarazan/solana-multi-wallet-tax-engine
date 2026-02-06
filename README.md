# Solana Multi-Wallet Tax Engine

Reliable, audit-style tax and PnL calculator for Solana wallets.

This tool rebuilds balances and cost basis directly from raw on-chain transactions instead of guessing prices or relying on portfolio trackers. It tracks FIFO lots across multiple wallets, handles swaps correctly, and exports tax-ready CSV reports.

Built because most crypto trackers lie. This one doesn’t.

---

## What this solves

Most trackers:
- guess prices
- break on swaps
- lose cost basis during transfers
- show fake PnL

This engine:
- reconstructs history from chain data only
- preserves cost basis across wallets
- tracks FIFO lots correctly
- treats token↔token swaps as non-taxable
- realizes gains only on true disposals
- exports clean CSV for taxes

No manual balances.  
No assumptions.  
No magic.

---

## Features

- Multi-wallet support
- FIFO accounting
- Inter-wallet transfer detection
- Swap-aware (Jupiter/Raydium/Orca)
- SOL + stablecoin cash accounting
- Realized + unrealized PnL
- Deposit / withdrawal tracking
- Dead token handling
- Tax-ready CSV export
- Deterministic + reproducible

---

## Tech Stack

- Python
- Helius API
- Kraken price data
- Dexscreener / Jupiter pricing fallback
- CSV exports

---

## Installation

### Clone
```bash
git clone https://github.com/skarazan/solana-multi-wallet-tax-engine.git
cd solana-multi-wallet-tax-engine
```

### Install dependency
```bash
pip install requests
```

(only dependency required)

---

## Setup

Open:

```
cryoto_tax_report.py
```

Edit the config section:

```python
HELIUS_API_KEY = "your-api-key-here"

WALLETS = [
    "wallet1",
    "wallet2"
]

WALLET_NAMES = {
    "wallet1": "Main",
    "wallet2": "Trading"
}
```

Get a free API key from:
https://helius.xyz

---

## Run

```bash
python cryoto_tax_report.py
```

---

## Output

Generates:

```
./solana_tax_output_multi_wallet/multi_wallet_tax_report.csv
```

Includes:
- Summary PnL
- Realized trades
- Unrealized holdings
- Per-wallet stats
- Token breakdown
- Deposits / withdrawals
- Inter-wallet transfers
- Warnings

Ready for Excel or tax software.

---

## How it works

1. Fetch all transactions from Helius
2. Rebuild positions from genesis
3. Track lots (quantity + USD cost)
4. Move lots during transfers
5. Realize gains only when selling to SOL/stables
6. Price remaining holdings
7. Export CSV

Everything is derived directly from on-chain data.

---

## Philosophy

Core rule:

Never assume prices. Only calculate from actual cash flow.

Accuracy > pretty charts.

This is an accounting engine, not a dashboard.

---

## Use Cases

- Crypto taxes
- Wallet reconciliation
- Trade journaling
- Backtesting strategies
- Audit-style bookkeeping

---

## Security Note

Do NOT commit real API keys or wallet addresses.

Use placeholders or environment variables.

---


---

## License

MIT
