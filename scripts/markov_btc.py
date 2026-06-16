# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "yfinance", "hmmlearn", "scipy", "requests"]
# ///
"""
Markov Regime Daily Signal — BTC-USD → Telegram
Framework: Roan (@RohOnChain). Adapted for daily GitHub Actions delivery.

Matrix display order: Bull → Sideways → Bear (rows & cols)
Sends one MarkdownV2 message per run via Telegram Bot API.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
TICKER      = os.getenv("TICKER", "BTC-USD")
YEARS       = int(os.getenv("YEARS", "10"))
WINDOW      = int(os.getenv("WINDOW", "20"))
THRESHOLD   = float(os.getenv("THRESHOLD", "0.05"))
MIN_TRAIN   = int(os.getenv("MIN_TRAIN", "252"))
BOT_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID     = os.environ["TELEGRAM_CHAT_ID"]
ENABLE_HMM  = os.getenv("ENABLE_HMM", "true").lower() == "true"

# Internal state indices: 0=Bear  1=Sideways  2=Bull
# Display order:          Bull=2  Sideways=1  Bear=0  → reindex [2,1,0]
INTERNAL_STATES = ["Bear", "Sideways", "Bull"]
DISPLAY_ORDER   = [2, 1, 0]


# ── MarkdownV2 escaping ───────────────────────────────────────────────────────
# Apply to ALL dynamic plain-text outside backtick/pre/bold/italic spans.
# Full list of chars reserved by Telegram MarkdownV2:
_MDV2_RESERVED = r"\_*[]()~`>#+-=|{}.!"

def esc(text: str) -> str:
    """Escape a plain-text string for Telegram MarkdownV2."""
    for ch in _MDV2_RESERVED:
        text = text.replace(ch, f"\\{ch}")
    return text


# ── Data ──────────────────────────────────────────────────────────────────────
def fetch_close(ticker: str, years: int) -> pd.Series:
    end   = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = end - pd.DateOffset(years=years)
    for attempt in (1, 2):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            print(f"yfinance error attempt {attempt}: {exc}", file=sys.stderr)
            df = pd.DataFrame()
        if not df.empty:
            break
        if attempt == 1:
            print("yfinance empty — retrying in 30s", file=sys.stderr)
            time.sleep(30)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


# ── Core model ────────────────────────────────────────────────────────────────
def label_regimes(close: pd.Series) -> pd.Series:
    rolling_return = close.pct_change(WINDOW)
    labels = pd.Series(1, index=close.index, dtype=int)
    labels[rolling_return >  THRESHOLD] = 2   # Bull
    labels[rolling_return < -THRESHOLD] = 0   # Bear
    return labels.loc[rolling_return.notna()]


def build_transition_matrix(labels: pd.Series) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=float)
    arr = np.asarray(labels, dtype=int)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def walk_forward_backtest(close: pd.Series, labels: pd.Series) -> dict:
    daily_ret = close.pct_change().dropna()
    idx       = labels.index.intersection(daily_ret.index)
    lab       = np.asarray(labels.loc[idx], dtype=int)
    rets      = daily_ret.loc[idx].to_numpy(dtype=float)
    if len(lab) < MIN_TRAIN + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}
    counts = np.zeros((3, 3), dtype=float)
    for i in range(MIN_TRAIN - 1):
        counts[lab[i], lab[i + 1]] += 1.0
    strategy_returns = []
    for t in range(MIN_TRAIN, len(lab) - 1):
        safe    = np.where(counts.sum(axis=1, keepdims=True) == 0, 1.0,
                           counts.sum(axis=1, keepdims=True))
        P_t     = counts / safe
        signal  = float(P_t[lab[t], 2] - P_t[lab[t], 0])
        strategy_returns.append(float(np.sign(signal)) * rets[t + 1])
        counts[lab[t - 1], lab[t]] += 1.0
    sr     = np.array(strategy_returns)
    std    = sr.std(ddof=1)
    sharpe = float(sr.mean() / std * np.sqrt(252)) if std > 0 else float("nan")
    equity = (1.0 + sr).cumprod()
    max_dd = float(((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min())
    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": len(sr)}


def fit_hmm(close: pd.Series):
    try:
        from hmmlearn import hmm as _hmm
    except Exception:
        return None
    X     = close.pct_change().dropna().to_numpy(dtype=float).reshape(-1, 1)
    model = _hmm.GaussianHMM(n_components=3, covariance_type="diag",
                              n_iter=200, random_state=42)
    model.fit(X)
    means = np.array([model.means_[k][0] for k in range(3)])
    order = np.argsort(means)
    names = ["Bear", "Sideways", "Bull"]
    return [(names[rank], int(k), float(means[k])) for rank, k in enumerate(order)]


# ── Message builder ───────────────────────────────────────────────────────────
def build_message(close: pd.Series, labels: pd.Series, P: np.ndarray,
                  pi: np.ndarray, bt: dict, hmm_regimes) -> str:
    P_d = P[np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]
    P2  = np.linalg.matrix_power(P, 2)
    P3  = np.linalg.matrix_power(P, 3)

    cs         = int(labels.iloc[-1])
    cs_label   = INTERNAL_STATES[cs]
    nd, d2, d3 = P[cs], P2[cs], P3[cs]

    signal    = float(nd[2] - nd[0])
    direction = (
        "\U0001f7e2 LONG"    if signal >= 0.3  else
        "\U0001f534 SHORT"   if signal <= -0.3 else
        "\U0001f7e1 NEUTRAL"
    )
    filled = int(abs(signal) * 10)
    bar    = "\u2588" * filled + "\u2591" * (10 - filled)

    last_price = float(close.iloc[-1])
    # esc() all dynamic plain-text — dates contain '-', ticker contains '-'
    last_date  = esc(str(close.index[-1].date()))
    run_date   = esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    ticker_esc = esc(TICKER)

    lines = [
        f"\U0001f4ca *{ticker_esc} Markov Daily Signal*",
        f"\U0001f5d3 {run_date}",
        f"\U0001f4b0 Last Close: `${last_price:,.2f}` \\(as of {last_date}\\)",
        "",
        "\u2501" * 21,
        f"\U0001f535 *Current Regime:* `{cs_label}`",
        f"\U0001f4e1 *Signal:* `{signal:+.4f}` \u2192 {direction}",
        bar,
        "",
        "\u2501" * 21,
        "\U0001f4d0 *3\u00d73 Transition Matrix \\(P\u00b9\\)*",
        "\\(Rows = Today \u2192 Cols = Tomorrow\\)",
        "",
        "```",
        "          Bull    Side    Bear",
        f"Bull     {P_d[0,0]*100:5.1f}%  {P_d[0,1]*100:5.1f}%  {P_d[0,2]*100:5.1f}%",
        f"Side     {P_d[1,0]*100:5.1f}%  {P_d[1,1]*100:5.1f}%  {P_d[1,2]*100:5.1f}%",
        f"Bear     {P_d[2,0]*100:5.1f}%  {P_d[2,1]*100:5.1f}%  {P_d[2,2]*100:5.1f}%",
        "```",
        "",
        "\U0001f512 *Persistence \\(diagonal\\):*",
        f"\u2022 \U0001f7e2 Bull  \u2192 Bull:  `{P_d[0,0]*100:.1f}%`",
        f"\u2022 \U0001f7e1 Side  \u2192 Side:  `{P_d[1,1]*100:.1f}%`",
        f"\u2022 \U0001f534 Bear  \u2192 Bear:  `{P_d[2,2]*100:.1f}%`",
        "",
        "\u2501" * 21,
        f"\U0001f52e *Multi\\-Step Forecast from* `{cs_label}`",
        "",
        "*1\\-Day \\(P\u00b9\\):*",
        f"\U0001f7e2 Bull `{nd[2]*100:.2f}%`  \U0001f7e1 Side `{nd[1]*100:.2f}%`  \U0001f534 Bear `{nd[0]*100:.2f}%`",
        "",
        "*2\\-Day \\(P\u00b2\\):*",
        f"\U0001f7e2 Bull `{d2[2]*100:.2f}%`  \U0001f7e1 Side `{d2[1]*100:.2f}%`  \U0001f534 Bear `{d2[0]*100:.2f}%`",
        "",
        "*3\\-Day \\(P\u00b3\\):*",
        f"\U0001f7e2 Bull `{d3[2]*100:.2f}%`  \U0001f7e1 Side `{d3[1]*100:.2f}%`  \U0001f534 Bear `{d3[0]*100:.2f}%`",
        "",
        "\U0001f4ca *Long\\-Run Stationary:*",
        f"Bull `{pi[2]*100:.1f}%`  Side `{pi[1]*100:.1f}%`  Bear `{pi[0]*100:.1f}%`",
        "",
        "\u2501" * 21,
        "\U0001f52c *HMM Regime Confirmation:*",
    ]

    if hmm_regimes:
        for lbl, k, m in hmm_regimes:
            emoji = (
                "\U0001f534" if lbl == "Bear"     else
                "\U0001f7e1" if lbl == "Sideways" else
                "\U0001f7e2"
            )
            lines.append(f"\u2022 {emoji} {lbl}: `{m*100:+.3f}%` mean daily return")
    else:
        lines.append("\u2022 _hmmlearn unavailable \u2014 skipped_")

    lines += [
        "",
        "\u2501" * 21,
        "\U0001f4c8 *Walk\\-Forward Backtest \\(10yr\\):*",
        f"\u2022 Sharpe \\(ann\\.\\): `{bt['sharpe']:.3f}`",
        f"\u2022 Max Drawdown:  `{bt['max_drawdown']*100:.2f}%`",
        f"\u2022 Trades:        `{bt['n_trades']:,}`",
    ]

    return "\n".join(lines)


# ── Telegram sender ───────────────────────────────────────────────────────────
def send_telegram(message: str) -> None:
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "MarkdownV2"}
    r    = requests.post(url, json=data, timeout=30)
    if not r.ok:
        print(f"Telegram error {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)
    print("\u2705 Telegram message sent.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Fetching {TICKER} ({YEARS}yr history)\u2026")
    close  = fetch_close(TICKER, YEARS)
    print(f"  {len(close)} rows | {close.index[0].date()} \u2192 {close.index[-1].date()}")
    labels = label_regimes(close)
    P      = build_transition_matrix(labels)
    pi     = stationary_distribution(P)
    bt     = walk_forward_backtest(close, labels)
    hmm_r  = fit_hmm(close) if ENABLE_HMM else None
    msg    = build_message(close, labels, P, pi, bt, hmm_r)
    send_telegram(msg)


if __name__ == "__main__":
    main()
