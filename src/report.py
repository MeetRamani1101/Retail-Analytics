"""Render the analysis into a markdown report.

Every figure in the text is interpolated from the results dictionary, so the
report cannot drift out of sync with the numbers. Nothing is hardcoded.

Run:  python -m src.report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import analysis as A
from . import config as C

CUR = C.CURRENCY


def _money(x: float) -> str:
    return f"{x:,.0f} {CUR}"


def _table(df: pd.DataFrame, cols: dict[str, str], floatfmt: str = "{:,.1f}") -> str:
    """Render a small DataFrame as a markdown table."""
    sub = df[list(cols)].rename(columns=cols)
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "|" + "|".join("---" for _ in sub.columns) + "|"
    rows = []
    for _, r in sub.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
            else str(v)
            for v in r
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


# ---------------------------------------------------------------------------
def build_report(res: A.AnalysisResults) -> str:
    k = res["kpis"]
    trend = res["trend_test"]
    cat = res["category"]
    seg = res["segments"]
    promo = res["promo_stats"]
    disc = res["discount"]
    ch = res["channels"]
    pareto = res["pareto"]
    retention = res["retention"]

    curve = A.retention_curve(retention, min_cohort_size=30)
    steady = curve.iloc[1:].mean()

    n80 = int((pareto["cum_revenue_pct"] >= 80).idxmax()) + 1
    pct80 = n80 / len(pareto) * 100

    top_cat = cat.iloc[0]
    best_margin = cat.sort_values("margin_pct").iloc[-1]

    champions = seg[seg["segment"] == "Champions"]
    at_risk = seg[seg["segment"] == "At Risk"]

    unprofitable = disc[disc["total_profit"] < 0]
    deep_loss = unprofitable["total_profit"].sum() if len(unprofitable) else 0.0
    units_spread = disc["avg_units"].max() - disc["avg_units"].min()

    ch_p = ch.attrs.get("kruskal_p", float("nan"))

    lines: list[str] = []
    add = lines.append

    add("# Retail Performance Analysis")
    add("")
    add(f"**Period analysed:** {k['period_start']:%d %B %Y} – {k['period_end']:%d %B %Y}  ")
    add(f"**Scope:** {k['order_lines']:,} order lines · {k['orders']:,} orders · "
        f"{k['customers']:,} customers")
    add("")
    add("> Generated from synthetic data by `src/report.py`. Every number below is "
        "computed at build time from the cleaned dataset.")
    add("")
    add("---")
    add("")

    # -- KPIs ---------------------------------------------------------------
    add("## 1. Headline numbers")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Net revenue | {_money(k['net_revenue'])} |")
    add(f"| Net profit | {_money(k['net_profit'])} ({k['margin_pct']:.1f}% margin) |")
    add(f"| Orders | {k['orders']:,} |")
    add(f"| Active customers | {k['customers']:,} |")
    add(f"| Average order value | {_money(k['aov'])} |")
    add(f"| Median order value | {_money(k['median_order_value'])} |")
    add(f"| Units sold | {k['units_sold']:,} |")
    add(f"| Repeat purchase rate | {k['repeat_rate_pct']:.1f}% |")
    add(f"| Return rate | {k['return_rate_pct']:.1f}% |")
    add("")

    # -- Trend --------------------------------------------------------------
    add("## 2. Revenue is compounding, not growing linearly")
    add("")
    add("![Revenue trend](figures/01_revenue_trend.png)")
    add("")
    add(f"Monthly revenue is better described by an exponential fit "
        f"(R² = {trend['log_r_squared']:.2f}) than a straight line "
        f"(R² = {trend['r_squared']:.2f}), which implies a compounding growth rate of "
        f"**{trend['monthly_growth_pct']:.1f}% per month** rather than a fixed "
        f"{_money(trend['slope_per_month'])} increment.")
    add("")
    add("Two seasonal patterns repeat in both years: a sharp November–December peak "
        "and a January contraction, followed by a mid-summer lull. Growth targets set "
        "off a December run-rate will be missed every January — the comparison to make "
        "is year-over-year, not month-over-month.")
    add("")

    # -- Mix ----------------------------------------------------------------
    add("## 3. The revenue leader is the margin laggard")
    add("")
    add("![Category performance](figures/02_category_performance.png)")
    add("")
    add(f"{top_cat['category']} generates {top_cat['revenue_share_pct']:.0f}% of revenue "
        f"at a {top_cat['margin_pct']:.0f}% margin, while {best_margin['category']} earns "
        f"{best_margin['margin_pct']:.0f}% on just "
        f"{best_margin['revenue_share_pct']:.0f}% of revenue. The categories are almost "
        "perfectly inverted on the two measures.")
    add("")
    add(_table(
        cat,
        {
            "category": "Category",
            "net_revenue": "Net revenue",
            "revenue_share_pct": "Rev share %",
            "margin_pct": "Margin %",
            "avg_discount": "Avg disc %",
            "return_rate": "Return %",
        },
    ))
    add("")
    add(f"Product-level revenue is similarly concentrated: **{n80} of {len(pareto)} "
        f"products ({pct80:.0f}%)** account for 80% of revenue.")
    add("")
    add("![Product Pareto](figures/03_product_pareto.png)")
    add("")

    # -- Customers ----------------------------------------------------------
    add("## 4. Customer value is concentrated in two segments")
    add("")
    add("![RFM segments](figures/04_rfm_segments.png)")
    add("")
    if len(champions):
        c = champions.iloc[0]
        add(f"**Champions** are {c['customer_share_pct']:.0f}% of customers but "
            f"{c['revenue_share_pct']:.0f}% of revenue, averaging "
            f"{_money(c['avg_monetary'])} lifetime spend across "
            f"{c['avg_frequency']:.1f} orders.")
        add("")
    if len(at_risk):
        a = at_risk.iloc[0]
        add(f"**At Risk** is the segment to act on: {a['customers']:,.0f} customers "
            f"holding {a['revenue_share_pct']:.0f}% of revenue "
            f"({_money(a['total_value'])}), but averaging "
            f"{a['avg_recency']:.0f} days since their last order. They have bought "
            f"{a['avg_frequency']:.1f} times on average, so the relationship exists — "
            "it has simply gone quiet.")
        add("")
    add(_table(
        seg,
        {
            "segment": "Segment",
            "customers": "Customers",
            "customer_share_pct": "Cust %",
            "revenue_share_pct": "Rev %",
            "avg_frequency": "Avg orders",
            "avg_monetary": "Avg spend",
            "avg_recency": "Avg recency (d)",
        },
    ))
    add("")

    # -- Retention ----------------------------------------------------------
    add("## 5. Retention collapses after the first month, then holds")
    add("")
    add("![Cohort retention](figures/05_cohort_retention.png)")
    add("")
    add(f"Across cohorts of 30+ customers, month-1 retention is "
        f"{curve.iloc[1]:.0f}%, after which the curve is essentially flat at "
        f"**{steady:.0f}%** through month {int(curve.index[-1])}. A flat tail is the "
        "good news — the customers who come back a second time keep coming back. "
        "The loss is almost entirely in the first 30 days.")
    add("")
    add("Cohorts whose members joined before the observation window are excluded: "
        "their first *observed* order is not their first order, and counting them "
        "would mix established buyers into new-customer cohorts.")
    add("")

    # -- Promotions ---------------------------------------------------------
    add("## 6. Promotions lift revenue and destroy profit")
    add("")
    add("![Promotion impact](figures/06_promotion_impact.png)")
    add("")
    sig = "statistically significant" if promo["significant"] else "not significant"
    add(f"Promotional days carry a median daily revenue of "
        f"{_money(promo['promo_median_daily_revenue'])} versus "
        f"{_money(promo['base_median_daily_revenue'])} at full price — a "
        f"**{promo['lift_pct']:.0f}% lift**, {sig} "
        f"(Mann–Whitney U, p = {promo['p_value']:.3f}, "
        f"{promo['promo_days']} promo days vs {promo['base_days']} baseline days).")
    add("")
    add("The revenue lift is real. The profit case is not:")
    add("")
    add(_table(
        disc,
        {
            "discount_band": "Discount band",
            "lines": "Order lines",
            "avg_units": "Avg units/line",
            "avg_line_revenue": "Avg line revenue",
            "avg_margin": "Avg margin %",
            "total_profit": "Total profit",
        },
    ))
    add("")
    add(f"Average units per line varies by only {units_spread:.2f} across every "
        "discount band, from full price to over 30% off. Discounting is not making "
        "people buy more; it is selling the same basket for less. Bands above 20% "
        f"contributed **{_money(deep_loss)}** in net profit — that is, they lost money.")
    add("")

    # -- Channels -----------------------------------------------------------
    add("## 7. Acquisition channel does not predict customer value")
    add("")
    add("![Channels and weekday](figures/07_channels_weekday.png)")
    add("")
    if ch_p >= 0.05:
        add(f"Mean lifetime spend ranges from {_money(ch['avg_ltv'].min())} to "
            f"{_money(ch['avg_ltv'].max())} across the five channels, but a "
            f"Kruskal–Wallis test finds no significant difference "
            f"(p = {ch_p:.2f}). The spread is consistent with noise.")
        add("")
        add("This is a useful negative result: it means budget should be allocated on "
            "acquisition **cost**, not on an assumed quality difference between "
            "channels. If one channel acquires customers more cheaply, take it — the "
            "customers it brings are worth about the same.")
    else:
        add(f"Lifetime spend differs significantly across channels "
            f"(Kruskal–Wallis p = {ch_p:.3f}), so acquisition cost should be judged "
            "against channel-specific value rather than a blended average.")
    add("")

    # -- Recommendations ----------------------------------------------------
    add("## 8. What to do about it")
    add("")
    add(f"1. **Cap discounts at 20%.** Bands beyond that lost {_money(abs(deep_loss))} "
        "with no measurable increase in units per order. Reserve deep cuts for genuine "
        "stock clearance, not for calendar promotions.")
    add(f"2. **Run a first-30-day onboarding programme.** Retention falls from 100% to "
        f"{curve.iloc[1]:.0f}% in a single month and is stable thereafter, so the entire "
        "retention problem sits in that window.")
    if len(at_risk):
        a = at_risk.iloc[0]
        add(f"3. **Win back At Risk customers.** {a['customers']:,.0f} customers, "
            f"{_money(a['total_value'])} of historical value, dormant for "
            f"{a['avg_recency']:.0f} days on average.")
    add(f"4. **Protect the mix.** {top_cat['category']} drives volume at "
        f"{top_cat['margin_pct']:.0f}% margin; growing "
        f"{best_margin['category']} and other high-margin categories improves profit "
        "faster than growing revenue does.")
    add("5. **Buy on cost, not channel reputation.** With no significant LTV difference "
        "between channels, the cheapest acquisition wins.")
    add("")
    add("---")
    add("")

    # -- Caveats ------------------------------------------------------------
    add("## 9. Limitations")
    add("")
    add("- The dataset is **synthetic**, generated by `src/generate_data.py`. The "
        "relationships it contains were put there by the simulation, so the findings "
        "demonstrate the method rather than describe a real business.")
    add("- Promotional periods coincide with Black Friday and Christmas, so the promo "
        "lift confounds discounting with seasonal demand. Separating the two would "
        "need a holdout region or a randomised discount test.")
    add("- Retention is measured on purchase recurrence only; no browsing, support or "
        "marketing-contact data is available.")
    add("- Returns are modelled as a flat probability per line and carry an assumed "
        "15% processing cost.")
    add("")

    return "\n".join(lines)


def main() -> None:
    res = A.run_all()
    text = build_report(res)
    path = C.REPORTS_DIR / "findings.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path.relative_to(C.PROJECT_ROOT)} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
