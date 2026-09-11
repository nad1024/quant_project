#!/usr/bin/env python3
"""
Institutional-grade quantitative edge analysis for the Momentum Trend Following playbook.
Maps live trade history against the book's Monte Carlo simulation benchmarks.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Playbook simulation benchmarks (Ch. 1) ──────────────────────────────────
BOOK_WIN_RATE = 0.30
BOOK_WIN_LOSS_RATIO_LOW = 3.0
BOOK_WIN_LOSS_RATIO_HIGH = 4.0
BOOK_MAX_CONSEC_LOSERS = 35
BOOK_AVG_MAX_DD = 0.198
BOOK_WORST_MAX_DD = 0.354
STARTING_EQUITY = 100_000.0
AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"

DATA_PATH = Path(__file__).parent / "trades.csv"
OUTPUT_CHART = Path(__file__).parent / "trading_edge_analysis.png"

# Market-cap brackets aligned with playbook case studies (low-float edge hypothesis)
MCAP_BRACKETS = [
    (0, 300_000_000, "Micro (<$300M)"),
    (300_000_000, 2_000_000_000, "Small ($300M–$2B)"),
    (2_000_000_000, 10_000_000_000, "Mid ($2B–$10B)"),
    (10_000_000_000, np.inf, "Large (>$10B)"),
]

HIGH_RISK_SECTORS = {"Healthcare", "Biotechnology"}


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load trades, coerce numerics, reconstruct equity where missing."""
    df = pd.read_csv(path)
    numeric_cols = [
        "entryPrice", "stopLoss", "riskPercent", "adrPercent", "marketCap",
        "posSize", "posValue", "tradeDuration", "netPnl", "tradeReturn",
        "rMultiple", "compEquity", "compNetPnl", "compTradeReturn",
        "compEquityChange",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["entryDate"] = pd.to_datetime(df["entryDate"], errors="coerce")
    df["exitDate"] = pd.to_datetime(df["exitDate"], errors="coerce")
    df["eventDate"] = df["exitDate"].fillna(df["entryDate"])

    # Reconstruct account equity at entry from position-sizing formula (Ch. 7)
    per_share_risk = (df["entryPrice"] - df["stopLoss"]).abs()
    risk_dollars = df["posSize"] * per_share_risk
    inferred_equity = np.where(
        (df["riskPercent"] > 0) & risk_dollars.notna(),
        risk_dollars / (df["riskPercent"] / 100.0),
        np.nan,
    )
    df["equityAtEntry"] = df["compEquity"].where(df["compEquity"].notna(), inferred_equity)
    df["posExposurePct"] = np.where(
        df["equityAtEntry"] > 0,
        (df["posValue"] / df["equityAtEntry"]) * 100.0,
        np.nan,
    )

    df["isWin"] = df["outcome"] == "Win"
    df["isLoss"] = df["outcome"] == "Loss"
    df["hasTrim"] = df["trims"].fillna("").astype(str).str.strip().ne("")
    df["setup"] = df["type"].fillna("Unknown")
    df["isHighRiskSector"] = (
        df["sector"].isin(HIGH_RISK_SECTORS)
        | df["industry"].str.contains("Biotech", case=False, na=False)
    )

    df["mcapBracket"] = pd.cut(
        df["marketCap"],
        bins=[b[0] for b in MCAP_BRACKETS] + [np.inf],
        labels=[b[2] for b in MCAP_BRACKETS],
        right=False,
    )

    return df.sort_values("eventDate").reset_index(drop=True)


def build_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological compounded equity from net PnL."""
    curve = df[["eventDate", "netPnl", "ticker", "outcome"]].copy()
    curve["compEquity"] = STARTING_EQUITY + curve["netPnl"].fillna(0).cumsum()
    curve["peakEquity"] = curve["compEquity"].cummax()
    curve["drawdown"] = (curve["compEquity"] - curve["peakEquity"]) / curve["peakEquity"]
    return curve


def max_consecutive_losses(df: pd.DataFrame) -> int:
    """Longest streak of consecutive losing trades in chronological order."""
    outcomes = df.sort_values("eventDate")["outcome"].tolist()
    max_streak = current = 0
    for o in outcomes:
        if o == "Loss":
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def profit_factor(df: pd.DataFrame) -> float:
    gross_win = df.loc[df["netPnl"] > 0, "netPnl"].sum()
    gross_loss = df.loc[df["netPnl"] < 0, "netPnl"].sum()
    if gross_loss == 0:
        return np.inf if gross_win > 0 else np.nan
    return gross_win / abs(gross_loss)


def compute_master_metrics(df: pd.DataFrame) -> dict:
    resolved = df[df["outcome"].isin(["Win", "Loss"])]
    wins = df[df["isWin"]]
    losses = df[df["isLoss"]]

    win_rate = len(wins) / len(resolved) if len(resolved) else np.nan
    loss_rate = len(losses) / len(resolved) if len(resolved) else np.nan
    avg_win_r = wins["rMultiple"].mean() if len(wins) else 0.0
    avg_loss_r = losses["rMultiple"].mean() if len(losses) else 0.0
    expectancy = (win_rate * avg_win_r) - (loss_rate * abs(avg_loss_r))

    book_mid_ratio = (BOOK_WIN_LOSS_RATIO_LOW + BOOK_WIN_LOSS_RATIO_HIGH) / 2
    book_expectancy = (BOOK_WIN_RATE * book_mid_ratio) - ((1 - BOOK_WIN_RATE) * 1.0)

    curve = build_equity_curve(df)
    max_dd = curve["drawdown"].min()

    return {
        "total_trades": len(df),
        "win_rate": win_rate * 100,
        "profit_factor": profit_factor(df),
        "net_pnl": df["netPnl"].sum(),
        "avg_r": df["rMultiple"].mean(),
        "avg_duration": df["tradeDuration"].mean(),
        "expectancy": expectancy,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "book_expectancy": book_expectancy,
        "book_win_rate": BOOK_WIN_RATE * 100,
        "book_win_loss_ratio": f"{BOOK_WIN_LOSS_RATIO_LOW:.0f}:1 – {BOOK_WIN_LOSS_RATIO_HIGH:.0f}:1",
        "max_drawdown": max_dd * 100,
        "max_consec_losses": max_consecutive_losses(df),
        "curve": curve,
    }


def adr_analysis(df: pd.DataFrame) -> dict:
    adr_df = df.dropna(subset=["adrPercent"])
    if adr_df.empty:
        return {"peak_adr": np.nan, "zero_wr_adr": np.nan, "quartile_table": pd.DataFrame()}

    adr_df = adr_df.copy()
    adr_df["adr_quartile"] = pd.qcut(adr_df["adrPercent"], 4, duplicates="drop")

    quartile_stats = (
        adr_df.groupby("adr_quartile", observed=True)
        .agg(
            trades=("id", "count"),
            win_rate=("isWin", "mean"),
            avg_r=("rMultiple", "mean"),
            net_pnl=("netPnl", "sum"),
            adr_min=("adrPercent", "min"),
            adr_max=("adrPercent", "max"),
        )
        .reset_index()
    )
    quartile_stats["win_rate"] *= 100

    # Fine-grained ADR bins for peak / zero win-rate detection
    bin_width = 1.0
    adr_min, adr_max = adr_df["adrPercent"].min(), adr_df["adrPercent"].max()
    bins = np.arange(np.floor(adr_min), np.ceil(adr_max) + bin_width, bin_width)
    adr_df["adr_bin"] = pd.cut(adr_df["adrPercent"], bins=bins, right=False)

    bin_stats = (
        adr_df.groupby("adr_bin", observed=True)
        .agg(trades=("id", "count"), win_rate=("isWin", "mean"))
        .reset_index()
    )
    bin_stats = bin_stats[bin_stats["trades"] >= 3]  # minimum sample for stability

    peak_row = bin_stats.loc[bin_stats["win_rate"].idxmax()] if len(bin_stats) else None
    peak_adr = (
        (peak_row["adr_bin"].left + peak_row["adr_bin"].right) / 2
        if peak_row is not None
        else np.nan
    )

    zero_bins = bin_stats[bin_stats["win_rate"] == 0]
    zero_wr_adr = (
        (zero_bins["adr_bin"].apply(lambda x: x.left).min() + zero_bins["adr_bin"].apply(lambda x: x.right).max()) / 2
        if len(zero_bins)
        else np.nan
    )

    return {
        "peak_adr": peak_adr,
        "zero_wr_adr": zero_wr_adr,
        "quartile_table": quartile_stats,
        "bin_stats": bin_stats,
    }


def sector_audit(df: pd.DataFrame) -> pd.DataFrame:
    def avg_loss_r(series: pd.Series) -> float:
        losses = series[df.loc[series.index, "isLoss"]]
        return losses.mean() if len(losses) else np.nan

    sector = (
        df.groupby("sector", dropna=False)
        .agg(
            trades=("id", "count"),
            win_rate=("isWin", lambda x: x.mean() * 100),
            avg_loss_r=("rMultiple", avg_loss_r),
            net_pnl=("netPnl", "sum"),
            avg_r=("rMultiple", "mean"),
        )
        .reset_index()
        .sort_values("net_pnl")
    )
    sector["sector"] = sector["sector"].fillna("(Unknown)")
    return sector


def risk_audit(df: pd.DataFrame) -> dict:
    avg_risk = df["riskPercent"].mean()
    risk_violations = int((df["riskPercent"] > 2.0).sum())
    below_min = int((df["riskPercent"] < 0.5).sum())

    exposure_valid = df["posExposurePct"].dropna()
    avg_exposure = exposure_valid.mean() if len(exposure_valid) else np.nan
    exposure_violations = int((df["posExposurePct"] > 50.0).sum())

    return {
        "avg_risk_pct": avg_risk,
        "risk_violations": risk_violations,
        "below_min_risk": below_min,
        "avg_exposure_pct": avg_exposure,
        "exposure_violations": exposure_violations,
    }


def trim_audit(df: pd.DataFrame) -> pd.DataFrame:
    trim_stats = (
        df.groupby("hasTrim")
        .agg(
            trades=("id", "count"),
            win_rate=("isWin", lambda x: x.mean() * 100),
            avg_r=("rMultiple", "mean"),
            net_pnl=("netPnl", "sum"),
        )
        .reset_index()
    )
    trim_stats["trim_status"] = trim_stats["hasTrim"].map(
        {True: "Partial Profit (Trim)", False: "Full Position (No Trim)"}
    )
    return trim_stats[["trim_status", "trades", "win_rate", "avg_r", "net_pnl"]]


def setup_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("setup")
        .agg(
            trades=("id", "count"),
            win_rate=("isWin", lambda x: x.mean() * 100),
            avg_r=("rMultiple", "mean"),
            net_pnl=("netPnl", "sum"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )


def print_markdown_table(headers: list[str], rows: list[list], title: str = "") -> None:
    """Pretty-print a markdown table to console."""
    if title:
        print(f"\n### {title}\n")

    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    print(fmt_row(headers))
    print(sep)
    for row in str_rows:
        print(fmt_row(row))


def format_num(val, decimals=2, suffix="", prefix="") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, float) and np.isinf(val):
        return "∞"
    if abs(val) >= 1_000_000:
        return f"{prefix}{val/1_000_000:,.{decimals}f}M{suffix}"
    if abs(val) >= 1_000:
        return f"{prefix}{val:,.{decimals}f}{suffix}"
    return f"{prefix}{val:.{decimals}f}{suffix}"


def plot_equity_and_drawdown(curve: pd.DataFrame, max_dd: float, max_streak: int) -> None:
    """Two-panel institutional chart."""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#f0f6fc")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    dates = curve["eventDate"]
    equity = curve["compEquity"]
    dd_pct = curve["drawdown"] * 100

    ax1.plot(dates, equity, color="#58a6ff", linewidth=1.8, label="Compounded Equity")
    ax1.fill_between(dates, STARTING_EQUITY, equity, alpha=0.15, color="#58a6ff")
    ax1.axhline(STARTING_EQUITY, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Account Equity ($)", fontsize=10)
    ax1.set_title(
        "Momentum Trend Following — Live Edge Analysis\n"
        "Playbook Simulation Benchmark vs. Actual Trade History",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

    final_eq = equity.iloc[-1]
    ret_pct = (final_eq / STARTING_EQUITY - 1) * 100
    ax1.annotate(
        f"Final: ${final_eq:,.0f}  ({ret_pct:+.1f}%)",
        xy=(dates.iloc[-1], final_eq),
        xytext=(-120, 15), textcoords="offset points",
        fontsize=9, color="#3fb950",
        arrowprops=dict(arrowstyle="->", color="#3fb950", lw=0.8),
    )

    ax2.fill_between(dates, dd_pct, 0, color="#f85149", alpha=0.55)
    ax2.plot(dates, dd_pct, color="#f85149", linewidth=1.2)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Trade Close Date", fontsize=10)
    ax2.axhline(max_dd, color="#ffa657", linestyle="--", linewidth=1,
                label=f"Max DD: {max_dd:.1f}%  |  Book sim avg: {BOOK_AVG_MAX_DD*100:.1f}%  |  Worst: {BOOK_WORST_MAX_DD*100:.1f}%")
    ax2.axhline(-BOOK_AVG_MAX_DD * 100, color="#8b949e", linestyle=":", linewidth=0.8, alpha=0.7)
    ax2.legend(loc="lower left", fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.text(
        0.99, 0.01,
        f"Max Consecutive Losses: {max_streak}  (Book simulation warning: {BOOK_MAX_CONSEC_LOSERS})",
        ha="right", va="bottom", fontsize=8, color="#8b949e",
    )

    fig.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n📊 Chart saved → {OUTPUT_CHART}")


def main() -> None:
    print("=" * 72)
    print("  MOMENTUM TREND FOLLOWING — QUANTITATIVE EDGE ANALYSIS")
    print("  Strategy: Breakout (BO) · Earnings Power (EP) · Pullback (PB)")
    print("  Filters: Market Regime (Setup #6) · Sector Rotation · Risk Caps")
    print("=" * 72)

    df = load_and_prepare(DATA_PATH)
    metrics = compute_master_metrics(df)
    adr = adr_analysis(df)
    sectors = sector_audit(df)
    risk = risk_audit(df)
    trims = trim_audit(df)
    setups = setup_breakdown(df)

    # ── 1. MASTER OVERVIEW ────────────────────────────────────────────────
    print_markdown_table(
        ["Metric", "Your Results", "Book Simulation", "Delta / Note"],
        [
            ["Total Trades", str(metrics["total_trades"]), "1,000 (sim basis)", "—"],
            ["Win Rate", f"{metrics['win_rate']:.1f}%", f"{metrics['book_win_rate']:.0f}%",
             f"{metrics['win_rate'] - metrics['book_win_rate']:+.1f}pp"],
            ["Profit Factor", format_num(metrics["profit_factor"]), "Driven by 3–4:1 W/L", "—"],
            ["Net PnL", format_num(metrics["net_pnl"], prefix="$"), "Varies (MC runs)", "—"],
            ["Avg R-Multiple", format_num(metrics["avg_r"]), "—", "—"],
            ["Avg Trade Duration", f"{metrics['avg_duration']:.1f} days", "—", "—"],
            ["Avg Win R", format_num(metrics["avg_win_r"]), f"~{BOOK_WIN_LOSS_RATIO_LOW:.0f}–{BOOK_WIN_LOSS_RATIO_HIGH:.0f}R", "—"],
            ["Avg Loss R", format_num(metrics["avg_loss_r"]), "−1.00R", "—"],
            ["Expectancy / Trade", f"{metrics['expectancy']:.3f}R", f"{metrics['book_expectancy']:.3f}R",
             f"{metrics['expectancy'] - metrics['book_expectancy']:+.3f}R"],
            ["Win : Loss Ratio", f"{abs(metrics['avg_win_r'] / metrics['avg_loss_r']):.2f}:1"
             if metrics["avg_loss_r"] else "N/A",
             metrics["book_win_loss_ratio"], "—"],
        ],
        title="1. MASTER OVERVIEW & EXPECTANCY",
    )

    # ── 2. SECRET SAUCE FILTERS ───────────────────────────────────────────
    if not adr["quartile_table"].empty:
        adr_rows = []
        for _, r in adr["quartile_table"].iterrows():
            adr_rows.append([
                str(r["adr_quartile"]),
                str(int(r["trades"])),
                f"{r['win_rate']:.1f}%",
                format_num(r["avg_r"]),
                format_num(r["net_pnl"], prefix="$"),
                f"{r['adr_min']:.1f}–{r['adr_max']:.1f}%",
            ])
        print_markdown_table(
            ["ADR Quartile", "Trades", "Win Rate", "Avg R", "Net PnL", "ADR Range"],
            adr_rows,
            title="2a. ADR% QUARTILE PERFORMANCE (Volatility Filter)",
        )
        print(f"  ▸ Peak win-rate ADR zone: ~{adr['peak_adr']:.1f}%")
        if not np.isnan(adr["zero_wr_adr"]):
            print(f"  ▸ Win rate drops to 0% near ADR: ~{adr['zero_wr_adr']:.1f}%")
        else:
            print("  ▸ No ADR bin (≥3 trades) with exactly 0% win rate detected")

    mcap_stats = (
        df.dropna(subset=["marketCap"])
        .groupby("mcapBracket", observed=True)
        .agg(trades=("id", "count"), win_rate=("isWin", lambda x: x.mean() * 100),
             avg_r=("rMultiple", "mean"), net_pnl=("netPnl", "sum"))
        .reset_index()
    )
    if not mcap_stats.empty:
        mcap_rows = [
            [str(r["mcapBracket"]), str(int(r["trades"])), f"{r['win_rate']:.1f}%",
             format_num(r["avg_r"]), format_num(r["net_pnl"], prefix="$")]
            for _, r in mcap_stats.iterrows()
        ]
        print_markdown_table(
            ["Market Cap", "Trades", "Win Rate", "Avg R", "Net PnL"],
            mcap_rows,
            title="2b. MARKET CAP BRACKETS (Low-Float Edge Hypothesis)",
        )

    sector_rows = [
        [str(r["sector"]), str(int(r["trades"])), f"{r['win_rate']:.1f}%",
         format_num(r["avg_loss_r"]), format_num(r["net_pnl"], prefix="$")]
        for _, r in sectors.iterrows()
    ]
    print_markdown_table(
        ["Sector", "Trades", "Win Rate", "Avg Loss R", "Net PnL"],
        sector_rows,
        title="2c. SECTOR AUDIT (Sector Rotation Model)",
    )

    hc = df[df["isHighRiskSector"]]
    hc_pnl = hc["netPnl"].sum()
    total_pnl = df["netPnl"].sum()
    hc_share = (hc_pnl / total_pnl * 100) if total_pnl != 0 else np.nan
    print(f"\n  ⚠️  Setup #5 — Healthcare/Biotech Flag:")
    print(f"      Trades: {len(hc)}  |  Net PnL: ${hc_pnl:,.2f}  |  "
          f"Share of total PnL: {hc_share:.1f}%")
    if hc_pnl < 0 and abs(hc_pnl) > abs(total_pnl) * 0.15:
        print("      → OUTSIZED DRAWDOWN CONTRIBUTOR — validates extreme-caution rule.")

    # ── 3. POSITION SIZING & RISK AUDIT ───────────────────────────────────
    print_markdown_table(
        ["Risk Metric", "Value", "Playbook Rule", "Status"],
        [
            ["Avg Risk / Trade", f"{risk['avg_risk_pct']:.2f}%", "0.5–2% (~1% typical)", "✓" if risk["avg_risk_pct"] <= 2 else "⚠"],
            ["Trades > 2% Risk", str(risk["risk_violations"]), "Never exceed 2%", "⚠ VIOLATION" if risk["risk_violations"] else "✓ Compliant"],
            ["Trades < 0.5% Risk", str(risk["below_min_risk"]), "Floor 0.5%", "Under-risked" if risk["below_min_risk"] else "—"],
            ["Avg Position Exposure", f"{risk['avg_exposure_pct']:.1f}%", "≤ 50% of equity", "✓" if risk["avg_exposure_pct"] <= 50 else "⚠"],
            ["Trades > 50% Exposure", str(risk["exposure_violations"]), "Hard cap 50%", "⚠ VIOLATION" if risk["exposure_violations"] else "✓ Compliant"],
        ],
        title="3. POSITION SIZING & RISK AUDIT",
    )

    trim_rows = [
        [r["trim_status"], str(int(r["trades"])), f"{r['win_rate']:.1f}%",
         format_num(r["avg_r"]), format_num(r["net_pnl"], prefix="$")]
        for _, r in trims.iterrows()
    ]
    print_markdown_table(
        ["Trim Status", "Trades", "Win Rate", "Avg R", "Net PnL"],
        trim_rows,
        title="3b. TRIM AUDIT (Scaled Exit vs. Full Hold)",
    )

    setup_rows = [
        [r["setup"], str(int(r["trades"])), f"{r['win_rate']:.1f}%",
         format_num(r["avg_r"]), format_num(r["net_pnl"], prefix="$")]
        for _, r in setups.iterrows()
    ]
    print_markdown_table(
        ["Setup Type", "Trades", "Win Rate", "Avg R", "Net PnL"],
        setup_rows,
        title="Setup Breakdown (BO / EP / PB)",
    )

    # ── 4. DRAWDOWN & EQUITY CURVE ────────────────────────────────────────
    print_markdown_table(
        ["Drawdown Metric", "Your Results", "Book Simulation"],
        [
            ["Maximum Drawdown", f"{metrics['max_drawdown']:.1f}%",
             f"Avg {BOOK_AVG_MAX_DD*100:.1f}% / Worst {BOOK_WORST_MAX_DD*100:.1f}%"],
            ["Max Consecutive Losses", str(metrics["max_consec_losses"]),
             str(BOOK_MAX_CONSEC_LOSERS)],
        ],
        title="4. DRAWDOWN & STREAK ANALYSIS",
    )

    plot_equity_and_drawdown(
        metrics["curve"], metrics["max_drawdown"], metrics["max_consec_losses"]
    )

    print("\n" + "=" * 72)
    print("  Analysis complete. Review tables above and trading_edge_analysis.png")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
