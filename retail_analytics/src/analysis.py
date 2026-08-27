"""Analytical layer.

Each function takes the clean order table and returns a tidy DataFrame (or a
dict of scalars). Nothing here plots or prints — that keeps the analysis
testable and lets `viz.py` and `report.py` consume the same numbers.

Run:  python -m src.analysis
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, cast

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .io_utils import load_table, save_table


class _LinregressResult(Protocol):
    """scipy.stats.linregress's return type isn't a public, typed class."""
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float


class TrendTest(TypedDict):
    slope_per_month: float
    r_squared: float
    p_value: float
    significant: bool
    log_r_squared: float
    monthly_growth_pct: float
    annualised_growth_pct: float
    better_fit: Literal["exponential", "linear"]


class AnalysisResults(TypedDict):
    df: pd.DataFrame
    kpis: dict[str, float]
    monthly: pd.DataFrame
    trend_test: TrendTest
    category: pd.DataFrame
    pareto: pd.DataFrame
    rfm: pd.DataFrame
    segments: pd.DataFrame
    retention: pd.DataFrame
    promo_summary: pd.DataFrame
    promo_stats: dict[str, float]
    discount: pd.DataFrame
    channels: pd.DataFrame
    weekday: pd.DataFrame


# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------
def kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    completed = df[df["status"] == "completed"]
    orders = completed.groupby("order_id")["net_revenue"].sum()

    span_days = (df["order_date"].max() - df["order_date"].min()).days or 1

    return {
        "period_start": df["order_date"].min(),
        "period_end": df["order_date"].max(),
        "order_lines": len(df),
        "orders": int(completed["order_id"].nunique()),
        "customers": int(df["customer_id"].nunique()),
        "net_revenue": float(df["net_revenue"].sum()),
        "net_profit": float(df["net_profit"].sum()),
        "margin_pct": float(df["net_profit"].sum() / max(df["net_revenue"].sum(), 1) * 100),
        "aov": float(orders.mean()),
        "median_order_value": float(orders.median()),
        "units_sold": int(completed["quantity"].sum()),
        "return_rate_pct": float(df["status"].eq("returned").mean() * 100),
        "revenue_per_day": float(df["net_revenue"].sum() / span_days),
        "repeat_rate_pct": float(
            (completed.groupby("customer_id")["order_id"].nunique() > 1).mean() * 100
        ),
    }


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df.groupby("order_month")
        .agg(
            net_revenue=("net_revenue", "sum"),
            net_profit=("net_profit", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_id", "nunique"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )
    m["aov"] = m["net_revenue"] / m["orders"]
    m["mom_growth_pct"] = m["net_revenue"].pct_change() * 100
    m["revenue_3mo_avg"] = m["net_revenue"].rolling(3, min_periods=1).mean()

    # Year-over-year, where a prior year exists.
    m["yoy_growth_pct"] = m["net_revenue"].pct_change(periods=12) * 100
    return m


def linear_trend_test(monthly: pd.DataFrame) -> TrendTest:
    """Fit both a linear and a log-linear trend to monthly revenue.

    Revenue series usually compound rather than grow by a fixed amount, so the
    log-linear fit (which yields a constant monthly growth *rate*) is reported
    alongside the straight line, and R² says which describes the data better.
    """
    y = monthly["net_revenue"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)

    lin = cast(_LinregressResult, stats.linregress(x, y))

    positive = y > 0
    log_fit = cast(_LinregressResult, stats.linregress(x[positive], np.log(y[positive])))
    monthly_growth = float(np.exp(log_fit.slope) - 1)

    return {
        "slope_per_month": float(lin.slope),
        "r_squared": float(lin.rvalue ** 2),
        "p_value": float(lin.pvalue),
        "significant": bool(lin.pvalue < 0.05),
        "log_r_squared": float(log_fit.rvalue ** 2),
        "monthly_growth_pct": monthly_growth * 100,
        "annualised_growth_pct": float(((1 + monthly_growth) ** 12 - 1) * 100),
        "better_fit": "exponential" if log_fit.rvalue ** 2 > lin.rvalue ** 2 else "linear",
    }


# ---------------------------------------------------------------------------
# Category and product performance
# ---------------------------------------------------------------------------
def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("category")
        .agg(
            net_revenue=("net_revenue", "sum"),
            net_profit=("net_profit", "sum"),
            orders=("order_id", "nunique"),
            units=("quantity", "sum"),
            avg_discount=("discount_pct", "mean"),
            return_rate=("status", lambda s: s.eq("returned").mean() * 100),
        )
        .reset_index()
    )
    cat["margin_pct"] = cat["net_profit"] / cat["net_revenue"] * 100
    cat["revenue_share_pct"] = cat["net_revenue"] / cat["net_revenue"].sum() * 100
    cat["revenue_per_order"] = cat["net_revenue"] / cat["orders"]
    return cat.sort_values("net_revenue", ascending=False).reset_index(drop=True)


def pareto_products(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    p = (
        df.groupby("product_id")
        .agg(net_revenue=("net_revenue", "sum"),
             net_profit=("net_profit", "sum"),
             units=("quantity", "sum"),
             category=("category", "first"))
        .sort_values("net_revenue", ascending=False)
        .reset_index()
    )
    p["cum_revenue_pct"] = p["net_revenue"].cumsum() / p["net_revenue"].sum() * 100
    p["rank"] = np.arange(1, len(p) + 1)
    p["product_pct"] = p["rank"] / len(p) * 100
    return p


# ---------------------------------------------------------------------------
# RFM segmentation
# ---------------------------------------------------------------------------
SEGMENT_RULES = [
    # (label, min_r_score, min_f_score, min_m_score)
    ("Champions",        4, 4, 4),
    ("Loyal",            3, 3, 3),
    ("Potential Loyal",  3, 2, 2),
    ("New / Promising",  4, 1, 1),
]


def _label_segment(r: int, f: int, m: int) -> str:
    for label, rmin, fmin, mmin in SEGMENT_RULES:
        if r >= rmin and f >= fmin and m >= mmin:
            return label
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and m >= 4:
        return "Can't Lose Them"
    if r <= 2:
        return "Hibernating"
    return "Needs Attention"


def rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["status"] == "completed"]
    snapshot = completed["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        completed.groupby("customer_id")
        .agg(
            recency_days=("order_date", lambda d: (snapshot - d.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("net_revenue", "sum"),
            first_order=("order_date", "min"),
            country=("country", "first"),
            channel=("acquisition_channel", "first"),
        )
        .reset_index()
    )
    rfm["tenure_days"] = (snapshot - rfm["first_order"]).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

    q = C.RFM_QUANTILES
    # Recency is inverted: fewer days since last purchase is better.
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"),
                             q, labels=range(q, 0, -1)).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"),
                             q, labels=range(1, q + 1)).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),
                             q, labels=range(1, q + 1)).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    rfm["segment"] = [
        _label_segment(r, f, m)
        for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"])
    ]
    return rfm


def segment_profile(rfm: pd.DataFrame) -> pd.DataFrame:
    prof = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_value=("monetary", "sum"),
        )
        .reset_index()
    )
    prof["customer_share_pct"] = prof["customers"] / prof["customers"].sum() * 100
    prof["revenue_share_pct"] = prof["total_value"] / prof["total_value"].sum() * 100
    return prof.sort_values("total_value", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cohort retention
# ---------------------------------------------------------------------------
def cohort_retention(df: pd.DataFrame, max_periods: int | None = None,
                     drop_left_censored: bool = True) -> pd.DataFrame:
    """Monthly retention matrix, cohorted on each customer's first purchase.

    Customers who joined before the observation window are excluded by default.
    Their first *observed* order is not their first order, so counting them
    would place established buyers in early cohorts and overstate retention.
    """
    max_periods = max_periods or C.COHORT_MAX_PERIODS
    completed = df[df["status"] == "completed"].copy()

    if drop_left_censored and "signup_date" in completed.columns:
        window_start = completed["order_date"].min().to_period("M").to_timestamp()
        completed = completed[completed["signup_date"] >= window_start]

    first = completed.groupby("customer_id")["order_date"].min()
    completed["cohort"] = (
        completed["customer_id"].map(first).dt.to_period("M")
    )
    order_period = completed["order_date"].dt.to_period("M")
    completed["period_index"] = (order_period - completed["cohort"]).apply(
        lambda x: x.n
    )

    counts = (
        completed.groupby(["cohort", "period_index"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    # Ensure one column per observable period, even if no cohort reached it.
    # Without this the matrix is ragged and a missing column is indistinguishable
    # from a period that genuinely saw no repeat purchases.
    last_period = completed["order_date"].dt.to_period("M").max()
    span = max((last_period - counts.index.min()).n, 0)
    full_cols = range(0, min(span, max_periods) + 1)
    counts = counts.reindex(columns=full_cols, fill_value=0)

    sizes = counts[0]
    retention = counts.divide(sizes, axis=0) * 100
    retention.index = retention.index.astype(str)

    # A cohort can only be observed for as many months as have elapsed since
    # it started; anything beyond that is not "0% retention", it is unknown.
    last_period = completed["order_date"].dt.to_period("M").max()
    for cohort_label in retention.index:
        elapsed = (last_period - pd.Period(cohort_label, freq="M")).n
        retention.loc[cohort_label, retention.columns > elapsed] = np.nan

    sizes.index = sizes.index.astype(str)
    retention.attrs["cohort_sizes"] = sizes
    return retention


def retention_curve(retention: pd.DataFrame, min_cohort_size: int = 30) -> pd.Series:
    """Average retention by month-since-first-order.

    Cohorts smaller than `min_cohort_size` are excluded: with only a handful of
    customers a single repeat purchase swings retention by tens of points, which
    would otherwise dominate the average.
    """
    sizes = retention.attrs.get("cohort_sizes")
    if sizes is not None:
        keep = sizes[sizes >= min_cohort_size].index
        retention = retention.loc[retention.index.isin(keep)]
    return retention.mean(axis=0, skipna=True)


# ---------------------------------------------------------------------------
# Promotion effectiveness
# ---------------------------------------------------------------------------
def promo_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    completed = df[df["status"] == "completed"].copy()

    daily = (
        completed.groupby(["order_date", "on_promo"])
        .agg(revenue=("net_revenue", "sum"),
             profit=("net_profit", "sum"),
             orders=("order_id", "nunique"))
        .reset_index()
    )

    promo_rev = daily.loc[daily["on_promo"], "revenue"]
    base_rev = daily.loc[~daily["on_promo"], "revenue"]

    # Daily revenue is right-skewed, so use a rank-based test rather than a
    # t-test, and report the median as well as the mean.
    u_stat, p_val = cast(
        "tuple[float, float]",
        stats.mannwhitneyu(promo_rev, base_rev, alternative="greater"),
    )

    summary = (
        completed.groupby("on_promo")
        .agg(
            lines=("order_id", "size"),
            revenue=("net_revenue", "sum"),
            profit=("net_profit", "sum"),
            units=("quantity", "sum"),
            avg_discount=("discount_pct", "mean"),
            avg_line_value=("net_revenue", "mean"),
        )
        .reset_index()
    )
    summary["margin_pct"] = summary["profit"] / summary["revenue"] * 100
    summary["label"] = np.where(summary["on_promo"], "Promotional", "Full price")

    stats_out = {
        "promo_median_daily_revenue": float(promo_rev.median()),
        "base_median_daily_revenue": float(base_rev.median()),
        "lift_pct": float((promo_rev.median() / base_rev.median() - 1) * 100),
        "mannwhitney_u": float(u_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "promo_days": int(daily["on_promo"].sum()),
        "base_days": int((~daily["on_promo"]).sum()),
    }
    return summary, stats_out


def discount_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """Units sold per order line across discount bands, by category."""
    completed = df[df["status"] == "completed"].copy()
    bands = pd.cut(completed["discount_pct"],
                   bins=[-0.1, 0.1, 10, 20, 30, 100],
                   labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"])
    completed["discount_band"] = bands

    out = (
        completed.groupby("discount_band", observed=True)
        .agg(lines=("quantity", "size"),
             avg_units=("quantity", "mean"),
             avg_line_revenue=("net_revenue", "mean"),
             avg_margin=("margin_pct", "mean"),
             total_profit=("net_profit", "sum"))
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# Acquisition channel value
# ---------------------------------------------------------------------------
def channel_value(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    ch = (
        rfm.groupby("channel")
        .agg(customers=("customer_id", "count"),
             avg_ltv=("monetary", "mean"),
             median_ltv=("monetary", "median"),
             avg_orders=("frequency", "mean"),
             total_value=("monetary", "sum"))
        .reset_index()
    )
    ch["value_share_pct"] = ch["total_value"] / ch["total_value"].sum() * 100

    # Is mean LTV genuinely different across channels?
    groups = [g["monetary"].to_numpy() for _, g in rfm.groupby("channel")]
    h_stat, p_val = stats.kruskal(*groups)
    ch.attrs["kruskal_h"] = float(h_stat)
    ch.attrs["kruskal_p"] = float(p_val)
    return ch.sort_values("total_value", ascending=False).reset_index(drop=True)


def weekday_effect(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    daily = (
        df[df["status"] == "completed"]
        .groupby(["order_date", "weekday"])["net_revenue"].sum()
        .reset_index()
    )
    out = (
        daily.groupby("weekday")["net_revenue"]
        .agg(mean_revenue="mean", median_revenue="median", days="size")
        .reindex(order)
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
def run_all(df: pd.DataFrame | None = None) -> AnalysisResults:
    if df is None:
        df = load_table(C.CLEAN_ORDERS)

    monthly = monthly_trend(df)
    rfm = rfm_segments(df)
    promo_summary, promo_stats = promo_analysis(df)

    results: AnalysisResults = {
        "df": df,
        "kpis": kpi_summary(df),
        "monthly": monthly,
        "trend_test": linear_trend_test(monthly),
        "category": category_performance(df),
        "pareto": pareto_products(df),
        "rfm": rfm,
        "segments": segment_profile(rfm),
        "retention": cohort_retention(df),
        "promo_summary": promo_summary,
        "promo_stats": promo_stats,
        "discount": discount_elasticity(df),
        "channels": channel_value(df, rfm),
        "weekday": weekday_effect(df),
    }
    return results


def main() -> None:
    res = run_all()
    k = res["kpis"]
    print(f"Revenue      : {k['net_revenue']:>14,.0f} {C.CURRENCY}")
    print(f"Profit       : {k['net_profit']:>14,.0f} {C.CURRENCY} "
          f"({k['margin_pct']:.1f}% margin)")
    print(f"Orders       : {k['orders']:>14,}")
    print(f"Customers    : {k['customers']:>14,}")
    print(f"AOV          : {k['aov']:>14,.2f} {C.CURRENCY}")
    print(f"Repeat rate  : {k['repeat_rate_pct']:>13.1f}%")

    save_table(res["rfm"], C.RFM_TABLE)
    res["retention"].to_csv(C.COHORT_TABLE)
    save_table(res["category"], C.PROCESSED_DIR / "category_performance.csv")
    save_table(res["segments"], C.PROCESSED_DIR / "segment_profile.csv")
    print(f"\nAnalysis tables written to {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
