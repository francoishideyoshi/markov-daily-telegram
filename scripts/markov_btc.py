# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "yfinance", "hmmlearn", "scipy", "requests"]
# ///
"""
Markov Regime Daily Signal — BTC-USD → Telegram
Framework: Roan (@RohOnChain). Adapted for daily GitHub Actions delivery.

parse_mode: HTML  (avoids all MarkdownV2 reserved-character escaping issues)
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

INTERNAL_STATES = ["Bear", "Sideways", "Bull"]
DISPLAY_ORDER   = [2, 1, 0]   # Bull, Sideways, Bear


# ── HTML helpers ──────────────────────────────────────────────────────────────
def b(t):    return f"<b>{t}</b>"
def code(t): return f"<code>{t}</code>"
def pre(t):  return f"<pre>{t}</pre>"
def esc(t):  return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

DIV = "━" * 21


# ── Data ──────────────────────────────────────────────────────────────────────
def fetch_close(ticker, years):
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
def label_regimes(close):
    rolling_return = close.pct_change(WINDOW)
    labels = pd.Series(1, index=close.index, dtype=int)
    labels[rolling_return >  THRESHOLD] = 2
    labels[rolling_return < -THRESHOLD] = 0
    return labels.loc[rolling_return.notna()]


def build_transition_matrix(labels):
    counts = np.zeros((3, 3), dtype=float)
    arr = np.asarray(labels, dtype=int)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def walk_forward_backtest(close, labels):
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
        safe   = np.where(counts.sum(axis=1, keepdims=True) == 0, 1.0,
                          counts.sum(axis=1, keepdims=True))
        P_t    = counts / safe
        signal = float(P_t[lab[t], 2] - P_t[lab[t], 0])
        strategy_returns.append(float(np.sign(signal)) * rets[t + 1])
        counts[lab[t - 1], lab[t]] += 1.0
    sr     = np.array(strategy_returns)
    std    = sr.std(ddof=1)
    sharpe = float(sr.mean() / std * np.sqrt(252)) if std > 0 else float("nan")
    equity = (1.0 + sr).cumprod()
    max_dd = float(((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min())
    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": len(sr)}


def fit_hmm(close):
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
def build_message(close, labels, P, pi, bt, hmm_regimes):
    P_d = P[np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]
    P2  = np.linalg.matrix_power(P, 2)
    P3  = np.linalg.matrix_power(P, 3)

    cs         = int(labels.iloc[-1])
    cs_label   = INTERNAL_STATES[cs]
    nd, d2, d3 = P[cs], P2[cs], P3[cs]

    signal    = float(nd[2] - nd[0])
    direction = (
        "🟢 LONG"    if signal >= 0.3  else
        "🔴 SHORT"   if signal <= -0.3 else
        "🟡 NEUTRAL"
    )
    filled = int(abs(signal) * 10)
    bar    = "█" * filled + "░" * (10 - filled)

    last_price = float(close.iloc[-1])
    last_date  = esc(str(close.index[-1].date()))
    run_date   = esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    ticker_esc = esc(TICKER)

    # Format price and signal outside f-strings to avoid nested quote issues
    price_str  = f"{last_price:,.2f}"
    signal_str = f"{signal:+.4f}"

    matrix_block = (
        "          Bull    Side    Bear\n"
        f"Bull     {P_d[0,0]*100:5.1f}%  {P_d[0,1]*100:5.1f}%  {P_d[0,2]*100:5.1f}%\n"
        f"Side     {P_d[1,0]*100:5.1f}%  {P_d[1,1]*100:5.1f}%  {P_d[1,2]*100:5.1f}%\n"
        f"Bear     {P_d[2,0]*100:5.1f}%  {P_d[2,1]*100:5.1f}%  {P_d[2,2]*100:5.1f}%"
    )

    lines = [
        f"📊 {b(ticker_esc + ' Markov Daily Signal')}",
        f"🗓 {run_date}",
        f"💰 Last Close: {code('$' + price_str)} (as of {last_date})",
        "",
        DIV,
        f"🔵 {b('Current Regime:')} {code(cs_label)}",
        f"📡 {b('Signal:')} {code(signal_str)} → {direction}",
        bar,
        "",
        DIV,
        f"📐 {b('3×3 Transition Matrix (P¹)')}",
        "(Rows = Today → Cols = Tomorrow)",
        "",
        pre(matrix_block),
        "",
        f"🔒 {b('Persistence (diagonal):')}",
        f"• 🟢 Bull  → Bull:  {code(str(round(P_d[0,0]*100, 1)) + '%')}",
        f"• 🟡 Side  → Side:  {code(str(round(P_d[1,1]*100, 1)) + '%')}",
        f"• 🔴 Bear  → Bear:  {code(str(round(P_d[2,2]*100, 1)) + '%')}",
        "",
        DIV,
        f"🔮 {b('Multi-Step Forecast from')} {code(cs_label)}",
        "",
        b("1-Day (P¹):"),
        f"🟢 Bull {code(str(round(nd[2]*100,2))+'%')}  🟡 Side {code(str(round(nd[1]*100,2))+'%')}  🔴 Bear {code(str(round(nd[0]*100,2))+'%')}",
        "",
        b("2-Day (P²):"),
        f"🟢 Bull {code(str(round(d2[2]*100,2))+'%')}  🟡 Side {code(str(round(d2[1]*100,2))+'%')}  🔴 Bear {code(str(round(d2[0]*100,2))+'%')}",
        "",
        b("3-Day (P³):"),
        f"🟢 Bull {code(str(round(d3[2]*100,2))+'%')}  🟡 Side {code(str(round(d3[1]*100,2))+'%')}  🔴 Bear {code(str(round(d3[0]*100,2))+'%')}",
        "",
        f"📊 {b('Long-Run Stationary:')}",
        f"Bull {code(str(round(pi[2]*100,1))+'%')}  Side {code(str(round(pi[1]*100,1))+'%')}  Bear {code(str(round(pi[0]*100,1))+'%')}",
        "",
        DIV,
        f"🔬 {b('HMM Regime Confirmation:')}",
    ]

    if hmm_regimes:
        for lbl, k, m in hmm_regimes:
            emoji   = "🔴" if lbl == "Bear" else ("🟡" if lbl == "Sideways" else "🟢")
            m_str   = ("+" if m >= 0 else "") + str(round(m * 100, 3)) + "%"
            lines.append(f"• {emoji} {lbl}: {code(m_str)} mean daily return")
    else:
        lines.append("• <i>hmmlearn unavailable — skipped</i>")

    # Extract bt values into plain variables — no nested quotes in f-strings
    sharpe_str   = str(round(bt["sharpe"], 3))
    drawdown_str = str(round(bt["max_drawdown"] * 100, 2)) + "%"
    trades_str   = f"{bt['n_trades']:,}"

    lines += [
        "",
        DIV,
        f"📈 {b('Walk-Forward Backtest (10yr):')}",
        f"• Sharpe (ann.):  {code(sharpe_str)}",
        f"• Max Drawdown:   {code(drawdown_str)}",
        f"• Trades:         {code(trades_str)}",
        "",
        "<i>⚠️ Backtests are historical. Not financial advice.</i>",
    ]

    return "\n".join(lines)


# ── Telegram sender ───────────────────────────────────────────────────────────
def send_telegram(message):
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    r    = requests.post(url, json=data, timeout=30)
    if not r.ok:
        print(f"Telegram error {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)
    print("✅ Telegram message sent.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Fetching {TICKER} ({YEARS}yr history)…")
    close  = fetch_close(TICKER, YEARS)
    print(f"  {len(close)} rows | {close.index[0].date()} → {close.index[-1].date()}")
    labels = label_regimes(close)
    P      = build_transition_matrix(labels)
    pi     = stationary_distribution(P)
    bt     = walk_forward_backtest(close, labels)
    hmm_r  = fit_hmm(close) if ENABLE_HMM else None
    msg    = build_message(close, labels, P, pi, bt, hmm_r)
    send_telegram(msg)


if __name__ == "__main__":
    main()
