# Solana Multi-Wallet Tax Engine

Reliable, audit-style tax and PnL calculator for Solana wallets.

This tool rebuilds balances and cost basis directly from raw on-chain transactions instead of guessing prices or relying on portfolio trackers. It tracks FIFO lots across multiple wallets, handles swaps correctly, and exports tax-ready CSV reports.

Built because most crypto trackers lie. This one doesn't.

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
- treats token-to-token swaps as taxable (US rules)
- derives fair market value from swap legs using conservation of value
- flags missing lot data instead of silently assuming $0 cost
- exports clean CSV for taxes

No manual balances.
No assumptions.
No magic.

---

## Features

- **Multi-wallet support** — track unlimited wallets simultaneously
- **FIFO accounting** — proper lot-level cost basis tracking with shortfall detection
- **Inter-wallet transfers** — auto-detects SOL/token transfers between your wallets, preserves cost basis
- **Swap-aware** — handles Jupiter, Raydium, Orca multi-hop routing
- **US tax compliant** — token-to-token swaps are taxable disposals
- **FMV derivation** — prices derived from swap legs (conservation of value), cached and validated
- **Airdrop detection** — non-cash inflows recorded as income at FMV
- **SOL + stablecoin + LST** — native SOL, USDC, USDT, jitoSOL, mSOL, bSOL
- **Realized + unrealized PnL** — complete picture with current market prices
- **Deposit / withdrawal tracking** — separates external flows from trading activity
- **Missing lot warnings** — flags trades where cost basis may be understated
- **Tax-ready CSV export** — detailed report with summary, trades, income, holdings, warnings
- **Helius API caching** — avoid redundant API calls on re-runs
- **Graceful error handling** — invalid wallets and API errors don't crash the whole run

---

## Architecture

```
├── config.py          # API keys, wallets, date range, mint addresses, thresholds
├── models.py          # Dataclasses (Lot, Sale, Position, Transfer, etc.) + helpers
├── helius.py          # Helius API client — tx fetching, balances, caching
├── prices.py          # PriceService — Kraken daily, DexScreener, Jupiter, FMV derivation
├── flows.py           # Parse Helius transactions into typed outflows/inflows
├── fifo.py            # FIFO engine — sell with shortfall tracking, transfer between wallets
├── pnl.py             # PnLEngine — core tax logic (purchases, sales, swaps, airdrops)
├── transfers.py       # Inter-wallet transfer detection + lot migration
├── deposits.py        # Deposit/withdrawal calculation
├── report.py          # CSV report generation
├── main.py            # Orchestrator — ties everything together
└── requirements.txt   # requests
```

---

## Tech Stack

- Python 3
- Helius Enhanced Transactions API
- Kraken OHLC for SOL/USD daily prices
- DexScreener + Jupiter for token pricing
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

Edit `config.py`:

```python
HELIUS_API_KEY = "your-api-key-here"

WALLETS = [
    "wallet1...",
    "wallet2...",
]

WALLET_NAMES = {
    "wallet1...": "Main",
    "wallet2...": "Trading",
}

START_DATE_UTC = "2025-01-01"
END_DATE_UTC = "2026-03-29"
```

Get a free API key from: https://helius.xyz

---

## Run

```bash
python main.py
```

---

## Output

Generates:

```
./solana_tax_output_multi_wallet/multi_wallet_tax_report.csv
```

Includes:
- Total P&L summary
- Per-wallet breakdown
- Realized trades with FIFO cost basis
- Income events (airdrops)
- Unrealized holdings with current prices
- Inter-wallet transfer log
- Deposit/withdrawal summary
- Warnings (missing lots, dust, etc.)

Ready for Excel or tax software.

---

## How it works

1. Fetch all transactions from Helius (with caching)
2. Validate wallet addresses, skip invalid ones gracefully
3. Extract exact SOL prices from SOL/USDC swap events
4. Detect inter-wallet transfers (SOL + tokens)
5. Pre-seed FIFO lots for received transfers (preserves cost basis)
6. Classify each transaction: purchase, sale, swap, airdrop
7. Run FIFO accounting — track shortfalls, flag missing data
8. Derive FMV from swap legs using conservation of value
9. Fetch current prices for unrealized positions
10. Calculate deposits and withdrawals
11. Export comprehensive CSV report

Everything is derived directly from on-chain data.

---

## Tax Logic

| Scenario | Treatment |
|---|---|
| SOL/USDC → Token | Purchase (cost basis = amount spent + fees) |
| Token → SOL/USDC | Sale (taxable, proceeds - fees - cost basis = gain/loss) |
| Token → Token | Swap disposal (taxable, both sides at FMV) |
| SOL received from own wallet | Transfer (not taxable, lots migrate with original cost) |
| Token airdrop (no outflow) | Income at FMV |
| SOL deposit from exchange | Deposit (lot at FMV, not income) |

---

## Philosophy

Core rule: **Never assume prices. Only calculate from actual cash flow.**

Accuracy over pretty charts. This is an accounting engine, not a dashboard.

---

## Use Cases

- Crypto taxes (US rules)
- Wallet reconciliation
- Trade journaling
- Audit-style bookkeeping

---

## Security Note

Do NOT commit real API keys or wallet addresses.

Edit `config.py` locally with your credentials — it's in `.gitignore` patterns for backup files.

---

## License

MIT
