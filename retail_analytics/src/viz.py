"""Chart generation.

Every function takes already-computed results from `analysis.py` and writes one
figure to `reports/figures/`. No analysis happens here, so a change to a metric
never has to be made in two places.

Run:  python -m src.viz
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from . import analysis as A
from . import config as C


# ---------------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": C.FIG_DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": "#4A5568",
        "axes.edgecolor": "#CBD5E0",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "xtick.color": "#4A5568",
        "ytick.color": "#4A5568",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save(fig: Figure, name: str) -> Path:
    path = C.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _thousands(x, _pos) -> str:
    """Compact axis labels that still distinguish nearby values.

    Below 10k a decimal is kept, otherwise ticks at 1.8k / 2.0k / 2.2k would
    all render as an indistinguishable "2k".
    """
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 10_000:
        return f"{x / 1_000:.0f}k"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}k"
    return f"{x:.0f}"


# ---------------------------------------------------------------------------
def plot_revenue_trend(monthly: pd.DataFrame, trend: A.TrendTest) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12},
    )

    x = monthly["order_month"]
    ax1.fill_between(x, monthly["net_revenue"], color=C.PALETTE["primary"], alpha=0.12)
    ax1.plot(x, monthly["net_revenue"], color=C.PALETTE["primary"],
             lw=2.2, marker="o", ms=4, label="Monthly net revenue")
    ax1.plot(x, monthly["revenue_3mo_avg"], color=C.PALETTE["secondary"],
             lw=1.8, ls="--", label="3-month moving average")

    # Plot whichever trend model actually describes the series better. A
    # straight line through a compounding series understates late growth and
    # can even dip below zero at the start.
    xi = np.arange(len(monthly), dtype=float)
    if trend["better_fit"] == "exponential":
        y = monthly["net_revenue"].to_numpy(dtype=float)
        mask = y > 0
        coef = np.polyfit(xi[mask], np.log(y[mask]), 1)
        fitted = np.exp(np.polyval(coef, xi))
        label = (f"Exponential trend, {trend['monthly_growth_pct']:.1f}%/mo "
                 f"(R²={trend['log_r_squared']:.2f})")
    else:
        fitted = np.polyval(np.polyfit(xi, monthly["net_revenue"], 1), xi)
        label = f"Linear trend (R²={trend['r_squared']:.2f})"

    ax1.plot(x, fitted, color=C.PALETTE["muted"], lw=1.6, ls=":", label=label)

    peak = monthly.loc[monthly["net_revenue"].idxmax()]
    ax1.annotate(
        f"Peak: {peak['net_revenue'] / 1000:,.0f}k {C.CURRENCY}\n"
        f"{peak['order_month']:%b %Y}",
        xy=(peak["order_month"], peak["net_revenue"]),
        xytext=(-70, -38), textcoords="offset points",
        fontsize=9, color=C.PALETTE["dark"],
        arrowprops=dict(arrowstyle="->", color=C.PALETTE["muted"], lw=1),
    )

    ax1.set_ylabel(f"Net revenue ({C.CURRENCY})")
    ax1.yaxis.set_major_formatter(_thousands)
    ax1.set_title("Revenue trend and month-over-month growth", loc="left")
    ax1.legend(loc="upper left", ncol=3, fontsize=9)

    growth = monthly["mom_growth_pct"].fillna(0)
    colors = np.where(growth >= 0, C.PALETTE["accent"], C.PALETTE["danger"])
    ax2.bar(x, growth, width=20, color=colors, alpha=0.85)
    ax2.axhline(0, color="#718096", lw=0.9)
    ax2.set_ylabel("MoM %")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    return _save(fig, "01_revenue_trend.png")


def plot_category(cat: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.28)

    colors = [C.CATEGORY_COLORS.get(c, C.PALETTE["primary"]) for c in cat["category"]]
    y = np.arange(len(cat))[::-1]

    ax1.barh(y, cat["net_revenue"], color=colors, alpha=0.9, height=0.65)
    ax1.set_yticks(y, cat["category"])
    ax1.xaxis.set_major_formatter(_thousands)
    ax1.set_xlabel(f"Net revenue ({C.CURRENCY})")
    ax1.set_title("Revenue is concentrated in Electronics", loc="left")
    for yi, (rev, share) in enumerate(zip(cat["net_revenue"], cat["revenue_share_pct"])):
        ax1.text(rev, y[yi], f"  {share:.0f}%", va="center", fontsize=9,
                 color=C.PALETTE["dark"])
    ax1.set_xlim(0, cat["net_revenue"].max() * 1.15)

    # Margin tells the opposite story to revenue.
    order = cat.sort_values("margin_pct")
    colors2 = [C.CATEGORY_COLORS.get(c, C.PALETTE["primary"]) for c in order["category"]]
    y2 = np.arange(len(order))
    ax2.barh(y2, order["margin_pct"], color=colors2, alpha=0.9, height=0.65)
    ax2.set_yticks(y2, order["category"])
    ax2.set_xlabel("Net margin (%)")
    ax2.set_title("...but margin runs the other way", loc="left")
    for yi, m in enumerate(order["margin_pct"]):
        ax2.text(m, yi, f"  {m:.0f}%", va="center", fontsize=9, color=C.PALETTE["dark"])
    ax2.set_xlim(0, order["margin_pct"].max() * 1.18)

    return _save(fig, "02_category_performance.png")


def plot_pareto(pareto: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=C.FIG_SIZE_WIDE)

    n = len(pareto)
    ax.bar(pareto["rank"], pareto["net_revenue"], color=C.PALETTE["primary"],
           alpha=0.55, width=0.9)
    ax.set_xlabel("Product rank by revenue")
    ax.set_ylabel(f"Net revenue ({C.CURRENCY})")
    ax.yaxis.set_major_formatter(_thousands)

    ax2 = ax.twinx()
    ax2.plot(pareto["rank"], pareto["cum_revenue_pct"],
             color=C.PALETTE["secondary"], lw=2.2)
    ax2.set_ylabel("Cumulative revenue (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(False)

    # Where do we cross 80% of revenue?
    idx = int((pareto["cum_revenue_pct"] >= 80).idxmax())
    r80 = cast(int, pareto.loc[idx, "rank"])
    ax2.axhline(80, color=C.PALETTE["muted"], ls="--", lw=1)
    ax2.axvline(r80, color=C.PALETTE["muted"], ls="--", lw=1)
    ax2.annotate(
        f"{r80} of {n} products ({r80 / n:.0%})\ndrive 80% of revenue",
        xy=(r80, 80), xytext=(28, -55), textcoords="offset points",
        fontsize=9.5, color=C.PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFF5EB", ec=C.PALETTE["secondary"],
                  lw=1),
        arrowprops=dict(arrowstyle="->", color=C.PALETTE["secondary"], lw=1.2),
    )

    ax.set_title("Product concentration (Pareto)", loc="left")
    return _save(fig, "03_product_pareto.png")


def plot_rfm(rfm: pd.DataFrame, segments: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6))
    fig.subplots_adjust(wspace=0.25)

    seg_order = segments["segment"].tolist()
    cmap = plt.get_cmap("tab10")
    seg_colors = {s: cmap(i % 10) for i, s in enumerate(seg_order)}

    for seg in seg_order:
        sub = rfm[rfm["segment"] == seg]
        ax1.scatter(sub["recency_days"], sub["frequency"],
                    s=np.clip(sub["monetary"] / 22, 6, 240),
                    color=seg_colors[seg], alpha=0.45,
                    edgecolors="white", linewidths=0.4, label=seg)

    ax1.set_xlabel("Recency — days since last order")
    ax1.set_ylabel("Frequency — number of orders")
    ax1.set_title("RFM landscape (bubble size = lifetime spend)", loc="left")
    ax1.legend(fontsize=8, loc="upper right", markerscale=0.6, ncol=2)

    # Customer share vs revenue share, side by side.
    y = np.arange(len(segments))[::-1]
    h = 0.38
    ax2.barh(y + h / 2, segments["customer_share_pct"], height=h,
             color=C.PALETTE["muted"], label="% of customers")
    ax2.barh(y - h / 2, segments["revenue_share_pct"], height=h,
             color=C.PALETTE["primary"], label="% of revenue")
    ax2.set_yticks(y, segments["segment"], fontsize=9)
    ax2.set_xlabel("Share (%)")
    ax2.set_title("A few segments carry the revenue", loc="left")
    ax2.legend(fontsize=9, loc="lower right")

    return _save(fig, "04_rfm_segments.png")


def plot_cohorts(retention: pd.DataFrame, min_size: int = 30) -> Path:
    sizes = retention.attrs.get("cohort_sizes")
    keep = sizes[sizes >= min_size].index if sizes is not None else retention.index
    mat = retention.loc[retention.index.isin(keep)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    fig.subplots_adjust(wspace=0.22)

    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#F7FAFC", "#BEE3F8", "#63B3ED", "#2B6CB0", "#1A365D"]
    )
    data = mat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    im = ax1.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=40)

    ax1.set_xticks(range(mat.shape[1]), mat.columns)
    ax1.set_yticks(range(mat.shape[0]), mat.index, fontsize=8.5)
    ax1.set_xlabel("Months since first purchase")
    ax1.set_ylabel("Acquisition cohort")
    ax1.set_title(f"Cohort retention (%) — cohorts of {min_size}+ customers", loc="left")
    ax1.grid(False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.2,
                     color="white" if v > 22 else "#2D3748")

    fig.colorbar(im, ax=ax1, shrink=0.8, label="Retention (%)")

    curve = A.retention_curve(retention, min_cohort_size=min_size)
    ax2.plot(curve.index, curve.to_numpy(), color=C.PALETTE["primary"],
             lw=2.4, marker="o", ms=5)
    ax2.fill_between(curve.index, curve.to_numpy(), color=C.PALETTE["primary"], alpha=0.12)
    ax2.set_xlabel("Months since first purchase")
    ax2.set_ylabel("Average retention (%)")
    ax2.set_xticks(list(curve.index)[::2])  # whole months only
    ax2.set_title("Retention flattens after month 1", loc="left")
    if len(curve) > 2:
        ax2.annotate(
            f"Settles near {curve.iloc[1:].mean():.0f}%",
            xy=(curve.index[3], curve.iloc[3]), xytext=(12, 42),
            textcoords="offset points", fontsize=9, color=C.PALETTE["dark"],
            arrowprops=dict(arrowstyle="->", color=C.PALETTE["muted"], lw=1),
        )

    return _save(fig, "05_cohort_retention.png")


def plot_promo(df: pd.DataFrame, discount: pd.DataFrame, promo_stats: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.25)

    completed = df[df["status"] == "completed"]
    daily = completed.groupby(["order_date", "on_promo"])["net_revenue"].sum().reset_index()
    groups = [daily.loc[~daily["on_promo"], "net_revenue"],
              daily.loc[daily["on_promo"], "net_revenue"]]

    bp = ax1.boxplot(groups, tick_labels=["Full price", "Promotional"],
                     patch_artist=True, widths=0.5, showfliers=False)
    for patch, col in zip(bp["boxes"], [C.PALETTE["muted"], C.PALETTE["secondary"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color(C.PALETTE["dark"])
        median.set_linewidth(1.8)

    ax1.set_ylabel(f"Daily net revenue ({C.CURRENCY})")
    ax1.yaxis.set_major_formatter(_thousands)
    sig = "p < 0.01" if promo_stats["p_value"] < 0.01 else f"p = {promo_stats['p_value']:.3f}"
    ax1.set_title(
        f"Promo days lift daily revenue {promo_stats['lift_pct']:.0f}% ({sig})",
        loc="left",
    )

    # Margin by discount band — the cost side of the same coin.
    bands = discount["discount_band"].astype(str)
    colors = [C.PALETTE["accent"] if v > 0 else C.PALETTE["danger"]
              for v in discount["total_profit"]]
    ax2.bar(bands, discount["total_profit"], color=colors, alpha=0.85, width=0.62)
    ax2.axhline(0, color="#718096", lw=1)
    ax2.set_ylabel(f"Total net profit ({C.CURRENCY})")
    ax2.set_xlabel("Discount band")
    ax2.yaxis.set_major_formatter(_thousands)
    ax2.set_title("Discounts beyond 20% destroy profit", loc="left")

    for i, (p, u) in enumerate(zip(discount["total_profit"], discount["avg_units"])):
        va = "bottom" if p >= 0 else "top"
        off = 6 if p >= 0 else -6
        ax2.annotate(f"{u:.2f} units/line", xy=(i, p), xytext=(0, off),
                     textcoords="offset points", ha="center", va=va, fontsize=8,
                     color=C.PALETTE["dark"])

    return _save(fig, "06_promotion_impact.png")


def plot_channels_weekday(channels: pd.DataFrame, weekday: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.25)

    ax1.bar(channels["channel"], channels["avg_ltv"], color=C.PALETTE["primary"],
            alpha=0.85, width=0.6)
    ax1.set_ylabel(f"Average lifetime spend ({C.CURRENCY})")
    ax1.tick_params(axis="x", rotation=20)
    p = channels.attrs.get("kruskal_p", float("nan"))
    verdict = "No significant difference" if p >= 0.05 else "Channels differ significantly"
    ax1.set_title(f"Lifetime value by channel\n{verdict} (Kruskal-Wallis p = {p:.2f})",
                  loc="left", fontsize=11.5)
    overall = np.average(channels["avg_ltv"], weights=channels["customers"])
    ax1.axhline(overall, color=C.PALETTE["danger"], ls="--", lw=1.2)
    ax1.annotate(f"overall mean {overall:,.0f}", xy=(0.02, overall),
                 xycoords=("axes fraction", "data"), xytext=(0, 6),
                 textcoords="offset points", fontsize=8.5, color=C.PALETTE["danger"])

    wk = weekday.copy()
    is_we = wk["weekday"].isin(["Saturday", "Sunday"])
    colors = np.where(is_we, C.PALETTE["secondary"], C.PALETTE["muted"])
    ax2.bar(wk["weekday"].str[:3], wk["mean_revenue"], color=colors, alpha=0.9, width=0.62)
    ax2.set_ylabel(f"Mean daily revenue ({C.CURRENCY})")
    ax2.yaxis.set_major_formatter(_thousands)
    wknd = wk.loc[is_we, "mean_revenue"].mean()
    wkdy = wk.loc[~is_we, "mean_revenue"].mean()
    ax2.set_title(f"Weekends run {wknd / wkdy - 1:+.0%} vs weekdays", loc="left")

    return _save(fig, "07_channels_weekday.png")


# ---------------------------------------------------------------------------
def build_all(res: A.AnalysisResults | None = None) -> list[Path]:
    apply_style()
    res = res or A.run_all()

    paths = [
        plot_revenue_trend(res["monthly"], res["trend_test"]),
        plot_category(res["category"]),
        plot_pareto(res["pareto"]),
        plot_rfm(res["rfm"], res["segments"]),
        plot_cohorts(res["retention"]),
        plot_promo(res["df"], res["discount"], res["promo_stats"]),
        plot_channels_weekday(res["channels"], res["weekday"]),
    ]
    return paths


def main() -> None:
    for p in build_all():
        print(f"  wrote {p.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
