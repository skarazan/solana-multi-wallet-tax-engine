# Solana Multi-Wallet Tax Engine

Accurate, audit-style tax and PnL calculator for Solana wallets.

This tool rebuilds balances and cost basis directly from raw on-chain transactions instead of guessing prices or relying on portfolio apps. It tracks FIFO lots across multiple wallets, handles swaps correctly, and produces tax-ready CSV reports.

Built because most crypto trackers lie. This one doesn’t.

---

## What it solves

Typical trackers:
- guess prices
- break on swaps
- lose cost basis during transfers
- show fake PnL

This engine:
- reconstructs history from genesis
- preserves cost basis across wallets
- treats swaps correctly (no fake taxable events)
- calculates realized gains only on real disposals
- outputs clean CSV for taxes

---

## Features

- Multi-wallet support
- FIFO lot accounting
- Inter-wallet transfer detection
- Token ↔ token swaps (non-taxable)
- SOL/stable disposals (taxable)
- Dead token loss handling
- Exact USD tracking (no price assumptions)
- Tax-ready CSV export
- Deterministic + reproducible results

---

## How it works

Core logic:

1. Fetch all transactions from Helius
2. Rebuild token positions from scratch
3. Track lots:
   - quantity
   - USD cost
   - wallet
   - date
4. Move lots during transfers
5. Realize PnL only when selling to SOL/stables
6. Export reports

No beginning balances.  
No manual inputs.  
No guessing.

Everything comes from chain data.

---

## Installation

### 1. Clone
```bash
git clone https://github.com/YOURNAME/solana-multi-wallet-tax-engine.git
cd solana-multi-wallet-tax-engine
```

### 2. Install deps
```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file:

```
HELIUS_API_KEY=your_key_here
```

Then edit:

```
config.py
```

Add your wallets:

```python
WALLETS = [
    "wallet1",
    "wallet2"
]
```

---

## Run

```bash
python main.py
```

Outputs:

```
reports/
  wallet_summary.csv
  realized_gains.csv
  transfers.csv
  holdings.csv
```

Ready for tax software.

---

## Example Output

| Date | Token | Proceeds | Cost Basis | Gain/Loss |
|------|---------|-----------|-------------|-----------|
| 2025-05-14 | BONK | 500 | 300 | +200 |

---

## Project Structure

```
.
├─ main.py
├─ config.py
├─ reports/
├─ src/
│  ├─ fetcher.py
│  ├─ pnl_engine.py
│  ├─ transfers.py
│  ├─ lots.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

---

## Philosophy

This tool follows one rule:

> Never assume prices. Only calculate from actual cash flow.

If something cannot be priced, it is treated as zero (loss), not guessed.

Accuracy > pretty charts.

---

## Use cases

- Taxes
- Audit-style accounting
- Wallet reconciliation
- Backtesting trading strategies
- Crypto bookkeeping

---

## Tech Stack

- Python
- Helius API
- CSV exports
- FIFO accounting

---

## License

MIT

---
