"""Tests for the analysis layer.

Hand-built frames with known answers, so a broken metric fails loudly instead
of returning a plausible-looking number.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    _label_segment,
    cohort_retention,
    kpi_summary,
    monthly_trend,
    linear_trend_test,
    pareto_products,
    retention_curve,
    rfm_segments,
)


def make_orders(rows):
    """Build a minimal clean-format order table from (customer, date, revenue)."""
    recs = []
    for i, (cust, date, rev) in enumerate(rows):
        d = pd.Timestamp(date)
        recs.append(
            {
                "order_id": f"O{i}",
                "customer_id": cust,
                "product_id": "P1",
                "category": "Electronics",
                "order_date": d,
                "signup_date": d,
                "quantity": 1,
                "revenue": rev,
                "cost": rev * 0.5,
                "profit": rev * 0.5,
                "net_revenue": rev,
                "net_profit": rev * 0.5,
                "status": "completed",
                "order_month": d.to_period("M").to_timestamp(),
                "discount_pct": 0.0,
                "on_promo": False,
                "acquisition_channel": "Email",
                "country": "Germany",
                "weekday": d.day_name(),
                "is_weekend": d.dayofweek >= 5,
            }
        )
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
class TestKpis:
    def test_totals_and_averages(self):
        df = make_orders([("C1", "2024-01-05", 100), ("C2", "2024-01-06", 300)])
        k = kpi_summary(df)
        assert k["net_revenue"] == pytest.approx(400)
        assert k["orders"] == 2
        assert k["customers"] == 2
        assert k["aov"] == pytest.approx(200)

    def test_returned_lines_excluded_from_revenue(self):
        df = make_orders([("C1", "2024-01-05", 100), ("C2", "2024-01-06", 300)])
        df.loc[1, ["status", "net_revenue"]] = ["returned", 0.0]
        k = kpi_summary(df)
        assert k["net_revenue"] == pytest.approx(100)
        assert k["return_rate_pct"] == pytest.approx(50.0)

    def test_repeat_rate(self):
        df = make_orders([("C1", "2024-01-05", 10), ("C1", "2024-02-05", 10),
                          ("C2", "2024-01-06", 10)])
        assert kpi_summary(df)["repeat_rate_pct"] == pytest.approx(50.0)


class TestTrend:
    def test_detects_a_real_linear_trend(self):
        rows = [("C1", f"2024-{m:02d}-05", 100 * m) for m in range(1, 13)]
        trend = linear_trend_test(monthly_trend(make_orders(rows)))
        assert trend["slope_per_month"] > 0
        assert trend["significant"]

    def test_prefers_exponential_for_compounding_series(self):
        rows = [("C1", f"2024-{m:02d}-05", 100 * (1.3 ** m)) for m in range(1, 13)]
        trend = linear_trend_test(monthly_trend(make_orders(rows)))
        assert trend["better_fit"] == "exponential"
        assert trend["monthly_growth_pct"] == pytest.approx(30, abs=1)

    def test_mom_growth_first_month_is_undefined(self):
        rows = [("C1", "2024-01-05", 100), ("C1", "2024-02-05", 150)]
        m = monthly_trend(make_orders(rows))
        assert pd.isna(m["mom_growth_pct"].iloc[0])
        assert m["mom_growth_pct"].iloc[1] == pytest.approx(50.0)


class TestPareto:
    def test_cumulative_share_reaches_one_hundred(self):
        df = make_orders([("C1", "2024-01-05", 100)])
        p = pareto_products(df)
        assert p["cum_revenue_pct"].iloc[-1] == pytest.approx(100.0)

    def test_ranked_descending(self):
        df = make_orders([("C1", "2024-01-05", 10), ("C2", "2024-01-06", 900)])
        df.loc[1, "product_id"] = "P2"
        p = pareto_products(df)
        assert p["product_id"].iloc[0] == "P2"


class TestSegmentLabels:
    def test_high_scores_are_champions(self):
        assert _label_segment(5, 5, 5) == "Champions"

    def test_lapsed_frequent_buyer_is_at_risk(self):
        assert _label_segment(1, 4, 3) == "At Risk"

    def test_lapsed_high_spender_is_cant_lose(self):
        assert _label_segment(1, 1, 5) == "Can't Lose Them"

    def test_every_combination_gets_a_label(self):
        for r in range(1, 6):
            for f in range(1, 6):
                for m in range(1, 6):
                    assert isinstance(_label_segment(r, f, m), str)


class TestRfm:
    def test_recency_is_inverted(self):
        """A recent buyer must score higher on R than a lapsed one."""
        rows = [(f"C{i}", "2024-12-01", 100) for i in range(50)]
        rows += [(f"D{i}", "2024-01-01", 100) for i in range(50)]
        rfm = rfm_segments(make_orders(rows)).set_index("customer_id")
        assert cast(float, rfm.loc["C0", "r_score"]) > cast(float, rfm.loc["D0", "r_score"])

    def test_frequency_counts_distinct_orders(self):
        rows = [("C1", "2024-01-05", 10), ("C1", "2024-03-05", 10)]
        rows += [(f"X{i}", "2024-02-01", 10) for i in range(20)]
        rfm = rfm_segments(make_orders(rows)).set_index("customer_id")
        assert rfm.loc["C1", "frequency"] == 2


class TestCohorts:
    def test_period_zero_is_always_full_retention(self):
        rows = [("C1", "2024-01-05", 10), ("C2", "2024-01-06", 10),
                ("C1", "2024-02-05", 10)]
        ret = cohort_retention(make_orders(rows))
        assert (ret[0].dropna() == 100.0).all()

    def test_known_retention_value(self):
        # Two customers start in January; one returns in February -> 50%.
        rows = [("C1", "2024-01-05", 10), ("C2", "2024-01-06", 10),
                ("C1", "2024-02-05", 10)]
        ret = cohort_retention(make_orders(rows))
        assert ret.loc["2024-01", 1] == pytest.approx(50.0)

    def test_unobserved_future_periods_are_nan_not_zero(self):
        """A cohort cannot have 0% retention in a month that has not happened."""
        rows = [("C1", "2024-01-05", 10), ("C2", "2024-02-06", 10)]
        ret = cohort_retention(make_orders(rows))
        # The February cohort has no month-1 yet, so it must be NaN.
        assert np.isnan(cast(float, ret.loc["2024-02", 1]))

    def test_small_cohorts_excluded_from_curve(self):
        rows = [("C1", "2024-01-05", 10)]  # cohort of one
        ret = cohort_retention(make_orders(rows))
        curve = retention_curve(ret, min_cohort_size=30)
        assert curve.isna().all() or curve.empty
