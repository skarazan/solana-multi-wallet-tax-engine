Solana Multi-Wallet Tax Engine
Reliable, audit-style tax and PnL calculator for Solana wallets.
This tool rebuilds balances and cost basis directly from raw on-chain transactions instead of guessing prices or trusting portfolio trackers.
It tracks FIFO lots across multiple wallets, handles swaps correctly, and exports tax-ready CSV reports.
Built because most crypto trackers lie. This one doesn’t.
What this does
Most trackers:
guess prices
break on swaps
lose cost basis during transfers
show fake PnL
This engine:
reconstructs history from chain data only
preserves cost basis across wallets
tracks FIFO lots correctly
treats token↔token swaps as non-taxable
realizes gains only on true disposals
exports clean CSV for taxes
No manual balances.
No assumptions.
No “magic”.
If it can’t be priced, it’s treated as zero (loss), not guessed.
Features
Multi-wallet support
FIFO accounting
Inter-wallet transfer detection
Swap-aware (Jupiter/Raydium/Orca)
SOL + stablecoin cash accounting
Dead token handling
Realized + unrealized PnL
Deposit / withdrawal tracking
Tax-ready CSV report
Deterministic + reproducible
Tech
Python
Helius API
Kraken price data
Dexscreener / Jupiter pricing fallback
CSV exports
Installation
1. Clone
git clone https://github.com/skarazan/solana-multi-wallet-tax-engine.git
cd solana-multi-wallet-tax-engine
2. Install dependencies
pip install requests
(only dependency needed)
Setup
Open:
cryoto_tax_report.py
Edit the config section:
HELIUS_API_KEY = "your-api-key-here"

WALLETS = [
    "wallet1",
    "wallet2"
]

WALLET_NAMES = {
    "wallet1": "Main",
    "wallet2": "Trading"
}
Get a free key from:
https://helius.xyz
Run
python cryoto_tax_report.py
Output
Creates:
./solana_tax_output_multi_wallet/multi_wallet_tax_report.csv
Includes:
Summary PnL
Realized trades
Unrealized holdings
Per-wallet stats
Token breakdown
Deposits / withdrawals
Inter-wallet transfers
Warnings
Ready to import into tax software or Excel.
How it works (simple)
Fetch all transactions from Helius
Rebuild positions from genesis
Track lots (qty + USD cost)
Move lots between wallets on transfers
Realize PnL only when selling to SOL/stables
Price remaining holdings
Export CSV
Everything is derived from chain data.
Philosophy
Core rule:
Never assume prices. Only calculate from actual cash flow.
Accuracy > pretty charts.
This is an accounting engine, not a dashboard.
Use cases
Crypto taxes
Wallet reconciliation
Trade journaling
Backtesting
Audit-style bookkeeping
Security note
Do NOT commit real API keys or wallets.
Use placeholders or environment variables.
