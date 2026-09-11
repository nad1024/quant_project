#!/usr/bin/env python3
"""
Live Alpha Scanner — Momentum Trend Following Playbook (Setups #1, #4, #6, Ch. 7)

Multi-layer pipeline:
  1. Market regime gate (QQQ/SPY 10/20 EMA)
  2. ADR% watchlist filter (quant audit: 4.0% – 26.0%)
  3. Daily coiling pattern match (Setup #1 checklist)
  4. Intraday 1-min trigger + position sizing (Setup #4 & Ch. 7)
  5. Tabulate dashboard output
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

# Suppress noisy yfinance / urllib log spam during batch downloads
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── Configuration (quant audit + playbook) ──────────────────────────────────
ACCOUNT_EQUITY = 100_000.0
RISK_PCT = 1.0                          # 1.0% per trade → $1,000 risk
MAX_EXPOSURE_PCT = 50.0                 # Ch. 7 hard cap
ADR_MIN = 4.0                           # Below playbook "orderly" threshold
ADR_MAX = 26.0                          # Above audit zero-win-rate zone (~32%)
COIL_MIN_DAYS = 3
COIL_MAX_DAYS = 15
VOLUME_EXPANSION_MULT = 1.5
REGIME_LOOKBACK_DAYS = 50
DAILY_LOOKBACK_DAYS = 60                # Extra buffer for EMA warm-up
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY_SEC = 2.0
BATCH_SLEEP_SEC = 0.35                  # Rate-limit courtesy between tickers

INDEX_TICKERS = ("QQQ", "SPY")

# High-momentum universe — liquid names across leading sectors/themes
WATCHLIST: list[str] = [
    # Semiconductors & hardware
    "NVDA", "AMD", "AVGO", "MRVL", "SMCI", "ARM", "MU", "LRCX", "KLAC", "AMAT",
    "AEHR", "RKLB", "PLTR", "IONQ", "RGTI", "QBTS", "QUBT", "SNDK", "LITE", "AAOI",
    # Software / AI
    "CRM", "NOW", "SNOW", "DDOG", "NET", "CRWD", "PANW", "ZS", "APP", "DUOL",
    # Crypto / fintech momentum
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "WULF", "IREN",
    # EV / auto / consumer cyclical
    "TSLA", "RIVN", "LCID", "CVNA", "UPST", "AFRM", "LMND", "CELH",
    # Industrials / materials / energy
    "PLTR", "SYM", "RKLB", "PL", "UEC", "UROY", "ALB", "CRML", "HYMC", "TMC",
    "AGX", "POWL", "ZIM", "FCEL", "CAR",
    # Communication / internet
    "META", "GOOGL", "AMZN", "NFLX", "RDDT", "SNAP", "PINS", "ROKU",
    # Leveraged / thematic ETFs (high ADR, liquid)
    "SOXL", "TQQQ", "LABU", "NAIL",
    # Additional momentum from trade history
    "GLW", "INTC", "APLD", "VELO", "TSEM", "NBIS", "NVTS", "AXTI", "GCT",
    "WOOF", "SEDG", "OPEN", "GME", "CVNA", "PHUN", "DRCT",
]

# Deduplicate while preserving order
WATCHLIST = list(dict.fromkeys(WATCHLIST))


@dataclass
class ScanResult:
    ticker: str
    adr_pct: float
    status: str = "—"
    entry_trigger: Optional[float] = None
    stop_loss: Optional[float] = None
    shares: int = 0
    position_value: float = 0.0
    notes: str = ""


# ── Data fetching with retry / rate-limit handling ──────────────────────────

def fetch_history(
    ticker: str,
    *,
    period: str = "3mo",
    interval: str = "1d",
    retries: int = API_RETRY_ATTEMPTS,
) -> Optional[pd.DataFrame]:
    """Download OHLCV with exponential backoff on failure."""
    for attempt in range(1, retries + 1):
        try:
            sink = StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                    timeout=15,
                )
            if raw is None or raw.empty:
                return None

            df = _normalize_ohlcv(raw)
            if df is not None and len(df) >= 5:
                return df
            return None
        except Exception as exc:
            if attempt == retries:
                print(f"  [!] {ticker}: fetch failed after {retries} attempts — {exc}")
                return None
            delay = API_RETRY_DELAY_SEC * attempt
            print(f"  [~] {ticker}: retry {attempt}/{retries} in {delay:.0f}s ({exc})")
            time.sleep(delay)
    return None


def _normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Flatten yfinance MultiIndex columns and standardize names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=rename)

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    df = df[list(required)].copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna(subset=["Close"])


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── Layer 1: Regime Gate (Setup #6) ─────────────────────────────────────────

def check_market_regime() -> bool:
    """
    Returns True if regime is bullish (at least one index has 10EMA >= 20EMA).
    Terminates with warning if BOTH indices are bearish.
    """
    print("\n" + "=" * 72)
    print("  LAYER 1 — MARKET REGIME GATE (Setup #6)")
    print("=" * 72)

    regime_rows = []
    bearish_count = 0

    for idx in INDEX_TICKERS:
        df = fetch_history(idx, period=f"{REGIME_LOOKBACK_DAYS}d", interval="1d")
        if df is None or len(df) < 25:
            print(f"  [!] Unable to load {idx} — treating as bearish (fail-safe).")
            bearish_count += 1
            regime_rows.append([idx, "N/A", "N/A", "UNKNOWN", "⚠"])
            continue

        df["ema10"] = ema(df["Close"], 10)
        df["ema20"] = ema(df["Close"], 20)
        e10, e20 = df["ema10"].iloc[-1], df["ema20"].iloc[-1]
        state = "BULLISH" if e10 >= e20 else "BEARISH"
        if state == "BEARISH":
            bearish_count += 1
        regime_rows.append([idx, f"{e10:.2f}", f"{e20:.2f}", state, "✓" if e10 >= e20 else "✗"])

    print(tabulate(
        regime_rows,
        headers=["Index", "10 EMA", "20 EMA", "Regime", "Gate"],
        tablefmt="rounded_outline",
    ))

    if bearish_count == len(INDEX_TICKERS):
        print("\n" + "!" * 72)
        print("  MARKET REGIME BEARISH: ALL BREAKOUT SIGNALS PAUSED")
        print("  Both $QQQ and $SPY have 10EMA < 20EMA. Standing down per Setup #6.")
        print("!" * 72 + "\n")
        return False

    print("\n  ✓ Regime gate OPEN — proceeding to watchlist scan.\n")
    return True


# ── Layer 2: ADR% Filter (Quant Audit) ────────────────────────────────────────

def calc_adr_pct(df: pd.DataFrame, window: int = 20) -> float:
    """20-day Average Daily Range as % of close."""
    if len(df) < window:
        return np.nan
    segment = df.tail(window)
    daily_range_pct = (segment["High"] - segment["Low"]) / segment["Close"] * 100.0
    return float(daily_range_pct.mean())


def adr_filter(ticker: str, df: pd.DataFrame) -> tuple[bool, float, str]:
    adr = calc_adr_pct(df)
    if np.isnan(adr):
        return False, adr, "insufficient data"
    if adr < ADR_MIN:
        return False, adr, f"ADR {adr:.1f}% < {ADR_MIN}% floor"
    if adr > ADR_MAX:
        return False, adr, f"ADR {adr:.1f}% > {ADR_MAX}% barcode zone"
    return True, adr, "pass"


# ── Layer 3: Pattern Matching — is_coiling() (Setup #1) ─────────────────────

def is_coiling(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Setup #1 checklist subset:
      - 10/20 EMA rising
      - Close surfing at or above both EMAs
      - 3–15 day sideways consolidation (compressed range vs. ADR)
    """
    min_bars = 30
    if len(df) < min_bars:
        return False, "insufficient history"

    df = df.copy()
    df["ema10"] = ema(df["Close"], 10)
    df["ema20"] = ema(df["Close"], 20)

    close = df["Close"].iloc[-1]
    e10, e20 = df["ema10"].iloc[-1], df["ema20"].iloc[-1]

    # Rising MAs (5-bar slope)
    ema10_rising = df["ema10"].iloc[-1] > df["ema10"].iloc[-6]
    ema20_rising = df["ema20"].iloc[-1] > df["ema20"].iloc[-6]
    if not (ema10_rising and ema20_rising):
        return False, "EMAs not rising"

    # Surfing the moving averages
    if close < e10 or close < e20:
        return False, "close below 10/20 EMA"

    # Prior thrust: 30%+ move in lookback window (playbook demand phase)
    lookback = 60
    if len(df) >= lookback:
        low_point = df["Close"].iloc[-lookback:-COIL_MAX_DAYS].min()
        if low_point > 0:
            thrust_pct = (close / low_point - 1) * 100
            if thrust_pct < 30:
                return False, f"insufficient prior thrust ({thrust_pct:.0f}%)"

    adr = calc_adr_pct(df)
    if np.isnan(adr) or adr <= 0:
        return False, "ADR unavailable"

    # Find best consolidation window (3–15 days) with compressed range
    best_ratio = np.inf
    best_window = 0
    for window in range(COIL_MIN_DAYS, COIL_MAX_DAYS + 1):
        segment = df.iloc[-window:]
        range_pct = (segment["High"].max() - segment["Low"].min()) / close * 100
        ratio = range_pct / adr
        if ratio < best_ratio:
            best_ratio = ratio
            best_window = window

    # Coiling = total range over window is < 2.5× single-day ADR (tight base)
    if best_ratio > 2.5:
        return False, f"range too wide ({best_ratio:.1f}× ADR over {best_window}d)"

    # All closes in consolidation window above both EMAs
    coil_seg = df.iloc[-best_window:]
    if not ((coil_seg["Close"] >= coil_seg["ema10"]) & (coil_seg["Close"] >= coil_seg["ema20"])).all():
        return False, "base not surfing MAs"

    # Volume contraction during base vs. thrust (lower vol = supply drying up)
    if len(df) >= 30:
        base_vol = coil_seg["Volume"].mean()
        thrust_vol = df.iloc[-30:-best_window]["Volume"].mean()
        if thrust_vol > 0 and base_vol > thrust_vol * 1.2:
            return False, "volume not contracting in base"

    return True, f"coiling {best_window}d (range {best_ratio:.1f}× ADR)"


# ── Layer 4: Intraday Trigger & Risk Sizing (Setup #4 & Ch. 7) ──────────────

def analyze_intraday(ticker: str, adr_pct: float) -> ScanResult:
    """Download 1-min data, detect trigger, compute position size."""
    result = ScanResult(ticker=ticker, adr_pct=adr_pct, status="WATCHING")

    intraday = fetch_history(ticker, period="1d", interval="1m")
    if intraday is None or len(intraday) < 5:
        result.status = "NO DATA"
        result.notes = "1-min data unavailable"
        return result

    first = intraday.iloc[0]
    opening_high = float(first["High"])
    opening_low = float(first["Low"])

    # LOD = session low so far (playbook stop)
    lod = float(intraday["Low"].min())
    current_price = float(intraday["Close"].iloc[-1])

    triggered = False
    for i in range(4, len(intraday)):
        candle = intraday.iloc[i]
        prev_three_vol = intraday.iloc[i - 3:i]["Volume"].mean()
        if prev_three_vol <= 0:
            continue
        breaks_opening_high = float(candle["High"]) > opening_high
        vol_expansion = float(candle["Volume"]) > VOLUME_EXPANSION_MULT * prev_three_vol
        if breaks_opening_high and vol_expansion:
            triggered = True
            break

    result.status = "TRIGGERED" if triggered else "WATCHING"
    result.entry_trigger = round(opening_high + 0.01, 2)  # first-candle high + tick
    result.stop_loss = round(lod, 2)

    # Ch. 7 position sizing
    per_share_risk = current_price - lod
    if per_share_risk <= 0:
        result.notes = "invalid risk (price ≤ LOD)"
        result.shares = 0
        result.position_value = 0.0
        return result

    risk_dollars = ACCOUNT_EQUITY * (RISK_PCT / 100.0)
    shares_by_risk = int(risk_dollars / per_share_risk)

    max_position_value = ACCOUNT_EQUITY * (MAX_EXPOSURE_PCT / 100.0)
    shares_by_exposure = int(max_position_value / current_price)

    shares = max(0, min(shares_by_risk, shares_by_exposure))
    position_value = shares * current_price

    result.shares = shares
    result.position_value = round(position_value, 2)

    exposure_pct = (position_value / ACCOUNT_EQUITY) * 100
    result.notes = (
        f"risk ${per_share_risk:.2f}/sh | "
        f"exposure {exposure_pct:.1f}%"
    )
    if shares_by_risk > shares_by_exposure:
        result.notes += " | capped at 50%"

    return result


def position_size_formula(
    current_price: float,
    lod: float,
) -> tuple[int, float, str]:
    """Standalone Ch. 7 sizing for testing."""
    per_share_risk = current_price - lod
    if per_share_risk <= 0:
        return 0, 0.0, "invalid risk"

    risk_dollars = ACCOUNT_EQUITY * (RISK_PCT / 100.0)
    shares_by_risk = int(risk_dollars / per_share_risk)
    max_pos = ACCOUNT_EQUITY * (MAX_EXPOSURE_PCT / 100.0)
    shares_by_exp = int(max_pos / current_price)
    shares = max(0, min(shares_by_risk, shares_by_exp))
    return shares, shares * current_price, ""


# ── Layer 5: Dashboard ───────────────────────────────────────────────────────

def print_dashboard(results: list[ScanResult]) -> None:
    print("\n" + "=" * 72)
    print("  LIVE ALPHA SCANNER — DASHBOARD")
    print(f"  Account: ${ACCOUNT_EQUITY:,.0f}  |  Risk: {RISK_PCT}%  |  "
          f"ADR Band: {ADR_MIN}%–{ADR_MAX}%  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    if not results:
        print("\n  No tickers passed all filters. Standing by.\n")
        return

    # Sort: TRIGGERED first, then by ADR descending
    status_order = {"TRIGGERED": 0, "WATCHING": 1, "NO DATA": 2}
    results.sort(key=lambda r: (status_order.get(r.status, 9), -r.adr_pct))

    rows = []
    for r in results:
        rows.append([
            r.ticker,
            f"{r.adr_pct:.1f}%",
            r.status,
            f"${r.entry_trigger:.2f}" if r.entry_trigger else "—",
            f"${r.stop_loss:.2f}" if r.stop_loss else "—",
            f"{r.shares:,}" if r.shares else "—",
            f"${r.position_value:,.0f}" if r.position_value else "—",
        ])

    print(tabulate(
        rows,
        headers=[
            "Ticker", "ADR%", "Status", "Entry Trigger", "Stop (LOD)",
            "Shares to Buy", "Position Value",
        ],
        tablefmt="rounded_outline",
        numalign="right",
        stralign="left",
    ))

    triggered = [r for r in results if r.status == "TRIGGERED"]
    watching = [r for r in results if r.status == "WATCHING"]
    print(f"\n  Summary: {len(triggered)} TRIGGERED  |  {len(watching)} WATCHING  |  "
          f"{len(results)} on coiling watchlist")
    print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_scanner(watchlist: Optional[list[str]] = None) -> list[ScanResult]:
    tickers = watchlist or WATCHLIST
    print(f"\n  Universe: {len(tickers)} tickers  |  "
          f"Filters: ADR {ADR_MIN}–{ADR_MAX}%  |  Coiling {COIL_MIN_DAYS}–{COIL_MAX_DAYS}d")

    adr_pass: list[tuple[str, float, pd.DataFrame]] = []
    coil_candidates: list[ScanResult] = []

    print("\n" + "=" * 72)
    print("  LAYER 2 — ADR% WATCHLIST FILTER")
    print("=" * 72)

    adr_rejected = 0
    for i, ticker in enumerate(tickers):
        df = fetch_history(ticker, period=f"{DAILY_LOOKBACK_DAYS}d", interval="1d")
        time.sleep(BATCH_SLEEP_SEC)

        if df is None:
            adr_rejected += 1
            continue

        passed, adr, reason = adr_filter(ticker, df)
        if passed:
            adr_pass.append((ticker, adr, df))
        else:
            adr_rejected += 1

    print(f"  ADR filter: {len(adr_pass)} passed / {adr_rejected} rejected")

    print("\n" + "=" * 72)
    print("  LAYER 3 — DAILY COILING PATTERN (Setup #1)")
    print("=" * 72)

    coil_rejected = 0
    for ticker, adr, df in adr_pass:
        coiling, reason = is_coiling(df)
        if coiling:
            coil_candidates.append(ScanResult(ticker=ticker, adr_pct=adr, notes=reason))
            print(f"  ✓ {ticker:<6} ADR {adr:5.1f}%  — {reason}")
        else:
            coil_rejected += 1

    print(f"\n  Coiling filter: {len(coil_candidates)} passed / {coil_rejected} rejected")

    if not coil_candidates:
        print_dashboard([])
        return []

    print("\n" + "=" * 72)
    print("  LAYER 4 — INTRADAY TRIGGER & RISK SIZING (Setup #4 / Ch. 7)")
    print("=" * 72)

    final_results: list[ScanResult] = []
    for candidate in coil_candidates:
        time.sleep(BATCH_SLEEP_SEC)
        scanned = analyze_intraday(candidate.ticker, candidate.adr_pct)
        scanned.notes = candidate.notes + " | " + scanned.notes
        final_results.append(scanned)
        icon = "🔥" if scanned.status == "TRIGGERED" else "👁"
        print(f"  {icon} {scanned.ticker:<6} {scanned.status:<10} "
              f"trigger ${scanned.entry_trigger}  stop ${scanned.stop_loss}  "
              f"shares {scanned.shares}")

    print_dashboard(final_results)
    return final_results


def main() -> int:
    print("=" * 72)
    print("  MOMENTUM TREND FOLLOWING — LIVE ALPHA SCANNER")
    print("  Playbook Setups #1 · #4 · #6  |  Ch. 7 Position Sizing")
    print("=" * 72)

    try:
        if not check_market_regime():
            return 1

        results = run_scanner()
        return 0 if results else 0

    except KeyboardInterrupt:
        print("\n  Scan interrupted by user.")
        return 130
    except Exception as exc:
        print(f"\n  [FATAL] Unhandled error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
