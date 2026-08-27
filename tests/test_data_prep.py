"""Tests for the cleaning layer.

These target the rules that are easy to get subtly wrong — date parsing across
mixed formats, country normalisation, and the price-typo repair — rather than
re-testing pandas itself.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    QualityReport,
    clean_orders,
    normalise_country,
    parse_mixed_dates,
)


# ---------------------------------------------------------------------------
class TestParseMixedDates:
    def test_parses_each_supported_format(self):
        s = pd.Series(["2024-03-15", "15/03/2024", "Mar 15, 2024"])
        out = parse_mixed_dates(s)
        assert out.notna().all()
        assert (out == pd.Timestamp("2024-03-15")).all()

    def test_day_first_is_not_read_as_month_first(self):
        # 13 cannot be a month, so a wrong parse would produce NaT or a shift.
        out = parse_mixed_dates(pd.Series(["13/07/2024"]))
        assert out.iloc[0] == pd.Timestamp("2024-07-13")

    def test_ambiguous_date_uses_day_first(self):
        # 05/03 is 5 March under the day-first convention the export uses.
        out = parse_mixed_dates(pd.Series(["05/03/2024"]))
        assert out.iloc[0] == pd.Timestamp("2024-03-05")

    def test_unparseable_becomes_nat_rather_than_raising(self):
        out = parse_mixed_dates(pd.Series(["not a date", "2024-01-01"]))
        assert pd.isna(out.iloc[0])
        assert out.iloc[1] == pd.Timestamp("2024-01-01")

    def test_earlier_parse_is_not_overwritten(self):
        s = pd.Series(["2024-01-02"] * 5)
        out = parse_mixed_dates(s)
        assert (out == pd.Timestamp("2024-01-02")).all()


class TestNormaliseCountry:
    def test_strips_whitespace_and_fixes_case(self):
        out = normalise_country(pd.Series(["  germany ", "FRANCE", "Spain"]))
        assert list(out) == ["Germany", "France", "Spain"]

    def test_expands_iso_codes(self):
        out = normalise_country(pd.Series(["DE", "NL", "  at "]))
        assert list(out) == ["Germany", "Netherlands", "Austria"]

    def test_collapses_variants_to_one_label(self):
        out = normalise_country(pd.Series(["Germany", "GERMANY", "DE", " germany "]))
        assert out.nunique() == 1


# ---------------------------------------------------------------------------
@pytest.fixture
def products():
    return pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "unit_cost": [40.0, 5.0],
            "list_price": [100.0, 20.0],
            "category": ["Electronics", "Beauty"],
        }
    )


def _order_row(**over):
    row = {
        "order_id": "O1",
        "order_line": 1,
        "order_date": "2024-05-01",
        "customer_id": "C1",
        "product_id": "P1",
        "quantity": 2,
        "unit_price": 100.0,
        "discount_pct": 0.0,
        "unit_cost": 40.0,
        "status": "completed",
    }
    row.update(over)
    return row


class TestCleanOrders:
    def test_drops_exact_duplicates(self, products):
        raw = pd.DataFrame([_order_row(), _order_row()])
        out = clean_orders(raw, products, QualityReport())
        assert len(out) == 1

    def test_repairs_decimal_point_typo(self, products):
        # 1000.0 on a product listed at 100.0 is a stray zero.
        raw = pd.DataFrame([_order_row(unit_price=1000.0)])
        out = clean_orders(raw, products, QualityReport())
        assert out["unit_price"].iloc[0] == pytest.approx(100.0)

    def test_leaves_legitimate_price_alone(self, products):
        raw = pd.DataFrame([_order_row(unit_price=120.0)])
        out = clean_orders(raw, products, QualityReport())
        assert out["unit_price"].iloc[0] == pytest.approx(120.0)

    def test_negative_quantity_becomes_positive_return(self, products):
        raw = pd.DataFrame([_order_row(quantity=-3, status="completed")])
        out = clean_orders(raw, products, QualityReport())
        assert out["quantity"].iloc[0] == 3
        assert out["status"].iloc[0] == "returned"

    def test_missing_unit_cost_recovered_from_product_master(self, products):
        raw = pd.DataFrame([_order_row(unit_cost=np.nan)])
        out = clean_orders(raw, products, QualityReport())
        assert out["unit_cost"].iloc[0] == pytest.approx(40.0)

    def test_missing_discount_recomputed_from_price(self, products):
        raw = pd.DataFrame([_order_row(unit_price=75.0, discount_pct=np.nan)])
        out = clean_orders(raw, products, QualityReport())
        assert out["discount_pct"].iloc[0] == pytest.approx(25.0)

    def test_revenue_and_profit_arithmetic(self, products):
        raw = pd.DataFrame([_order_row(quantity=2, unit_price=100.0, unit_cost=40.0)])
        out = clean_orders(raw, products, QualityReport()).iloc[0]
        assert out["revenue"] == pytest.approx(200.0)
        assert out["cost"] == pytest.approx(80.0)
        assert out["profit"] == pytest.approx(120.0)

    def test_returned_line_earns_no_net_revenue(self, products):
        raw = pd.DataFrame([_order_row(status="returned")])
        out = clean_orders(raw, products, QualityReport()).iloc[0]
        assert out["net_revenue"] == 0.0
        assert out["net_profit"] < 0  # processing cost, not profit

    def test_out_of_range_rows_removed(self, products):
        raw = pd.DataFrame([_order_row(), _order_row(order_id="O2", quantity=9999)])
        out = clean_orders(raw, products, QualityReport())
        assert len(out) == 1

    def test_outliers_are_flagged_not_deleted(self, products):
        rows = [_order_row(order_id=f"O{i}", unit_price=10.0, quantity=1)
                for i in range(60)]
        rows.append(_order_row(order_id="BIG", unit_price=100.0, quantity=50))
        out = clean_orders(pd.DataFrame(rows), products, QualityReport())
        assert len(out) == 61
        assert out.loc[out["order_id"] == "BIG", "is_outlier"].iloc[0]

    def test_quality_report_records_every_step(self, products):
        rep = QualityReport()
        clean_orders(pd.DataFrame([_order_row(), _order_row()]), products, rep)
        frame = rep.to_frame()
        assert "drop exact duplicates" in set(frame["step"])
        assert rep.rows_in == 2
        assert rep.rows_out == 1
