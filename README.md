# markov-daily-telegram

Daily BTC-USD Markov regime signal → Telegram, via GitHub Actions.

Framework: Roan (@RohOnChain). Adapted from [jackson-video-resources/markov-hedge-fund-method](https://github.com/jackson-video-resources/markov-hedge-fund-method).

## What it does

Runs every day at **08:00 UTC (16:00 HKT)**:

1. Downloads 10 years of BTC-USD daily closes via `yfinance`
2. Labels each day as **Bull / Sideways / Bear** (rolling 20-day return ±5%)
3. Builds the **3×3 Markov transition matrix** (MLE)
4. Computes **P¹ / P² / P³** multi-step forecasts from today's regime
5. Derives the **trading signal** = P(Bull) − P(Bear)
6. Runs a **walk-forward backtest** (no lookahead bias)
7. Fits a **Hidden Markov Model** for regime confirmation
8. Sends everything as a formatted **Telegram message**

## Matrix display order

Rows and columns follow **Bull → Sideways → Bear** (top-left = Bull/Bull persistence).

## Setup

### 1. Fork / clone this repo

### 2. Add GitHub Actions secrets

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret name           | Value                          |
|-----------------------|-------------------------------|
| `TELEGRAM_BOT_TOKEN`  | Your bot token from @BotFather |
| `TELEGRAM_CHAT_ID`    | Your chat / channel ID         |

### 3. Enable Actions

Actions are enabled by default on new repos. Trigger manually via
**Actions → Markov Daily Signal → Run workflow** to test.

## Configuration

Edit environment variables in `.github/workflows/daily.yml`:

| Variable     | Default   | Description                          |
|--------------|-----------|--------------------------------------|
| `TICKER`     | `BTC-USD` | yfinance ticker symbol               |
| `YEARS`      | `10`      | Years of history to download         |
| `WINDOW`     | `20`      | Rolling-return window (days)         |
| `THRESHOLD`  | `0.05`    | ±5% threshold for Bull/Bear labels   |
| `MIN_TRAIN`  | `252`     | Min training rows before backtest    |
| `ENABLE_HMM` | `true`    | Include Hidden Markov Model section  |

## Schedule

Cron: `0 8 * * *` → 08:00 UTC = 16:00 HKT.
Change in `.github/workflows/daily.yml` to suit your timezone.

## Dependencies

`numpy`, `pandas`, `yfinance`, `hmmlearn`, `scipy`, `requests`
All installed automatically by the workflow.

## Disclaimer

Backtests are historical, not forward-looking. Not financial advice.
