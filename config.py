"""Configuration constants for the Solana DeFi PnL calculator."""

# =========================
# API Keys
# =========================

HELIUS_API_KEY = "insert your free helius rpc api"

# =========================
# Wallets
# =========================

WALLETS = [
    # Add your Solana wallet addresses here, e.g.:
    # "YourSolanaWalletAddressHere...",
]

WALLET_NAMES = {
    # Map wallet addresses to friendly names, e.g.:
    # "YourSolanaWalletAddressHere...": "Main Wallet",
}

# =========================
# Date Range
# =========================

START_DATE_UTC = "2025-01-01"
END_DATE_UTC = "2026-03-29"

# =========================
# Helius Settings
# =========================

HELIUS_MIN_DELAY_SEC = 0.12
HELIUS_MAX_RETRIES = 8
HELIUS_PAGE_LIMIT = 100

# =========================
# Paths
# =========================

OUTPUT_DIR = "./solana_tax_output_multi_wallet"
CACHE_DIR = "./cache_helius_2025"
USE_CACHE = True

# =========================
# Token Mints
# =========================

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
STABLE_MINTS = {USDC_MINT, USDT_MINT}

SOL_MINT = "So11111111111111111111111111111111111111112"

JITO_SOL = "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn"
MSOL = "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"
BSOL = "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1"
LST_MINTS = {JITO_SOL, MSOL, BSOL}

# "Cash" = SOL + stablecoins + LSTs.
# Selling non-cash tokens TO these = taxable event.
CASH_MINTS = {SOL_MINT} | STABLE_MINTS | LST_MINTS

# =========================
# Thresholds
# =========================

DUST_USD = 0.50
