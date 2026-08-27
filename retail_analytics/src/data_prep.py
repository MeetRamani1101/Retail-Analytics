"""Load the raw exports, clean them, and produce an analysis-ready table.

Every transformation appends a line to a `QualityReport`, so the cleaning is
auditable rather than a black box: you can see exactly how many rows each rule
touched and why.

Run:  python -m src.data_prep
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import save_table

COUNTRY_FIXES = {
    "DE": "Germany",
    "GER": "Germany",
    "NL": "Netherlands",
    "FR": "France",
    "AT": "Austria",
    "ES": "Spain",
    "IT": "Italy",
}


# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Accumulates a human-readable audit trail of the cleaning run."""

    rows_in: int = 0
    rows_out: int = 0
    steps: list[tuple[str, int, str]] = field(default_factory=list)

    def log(self, step: str, n_affected: int, note: str = "") -> None:
        self.steps.append((step, n_affected, note))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "rows_affected", "note"])

    def render(self) -> str:
        lines = ["Data quality report", "=" * 62,
                 f"rows read    : {self.rows_in:,}",
                 f"rows retained: {self.rows_out:,} "
                 f"({self.rows_out / max(self.rows_in, 1):.1%})", "-" * 62]
        for step, n, note in self.steps:
            lines.append(f"{step:<34} {n:>7,}  {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Parse a date column that mixes ISO, day-first and month-name formats.

    Tried in order of specificity; each pass only fills values still missing,
    so an earlier successful parse is never overwritten.
    """
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%b %d, %Y", "%m/%d/%Y"):
        missing = out.isna()
        if not missing.any():
            break
        out.loc[missing] = pd.to_datetime(s[missing], format=fmt, errors="coerce")
    missing = out.isna()
    if missing.any():  # last resort: let pandas infer
        out.loc[missing] = pd.to_datetime(s[missing], errors="coerce", dayfirst=True)
    return out


def normalise_country(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.strip().str.title()
    codes = s.astype("string").str.strip().str.upper()
    return cleaned.mask(codes.isin(COUNTRY_FIXES), codes.map(COUNTRY_FIXES))


# ---------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(C.RAW_ORDERS, dtype={"order_id": "string",
                                              "customer_id": "string",
                                              "product_id": "string"})
    customers = pd.read_csv(C.RAW_CUSTOMERS, dtype={"customer_id": "string"})
    products = pd.read_csv(C.RAW_PRODUCTS, dtype={"product_id": "string"})
    return orders, customers, products


def clean_orders(orders: pd.DataFrame, products: pd.DataFrame,
                 rep: QualityReport) -> pd.DataFrame:
    df = orders.copy()
    rep.rows_in = len(df)

    # 1. Exact duplicates -----------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    rep.log("drop exact duplicates", before - len(df), "identical rows removed")

    # 2. Dates ----------------------------------------------------------------
    df["order_date"] = parse_mixed_dates(df["order_date"])
    bad_dates = df["order_date"].isna().sum()
    df = df[df["order_date"].notna()].copy()
    rep.log("drop unparseable dates", int(bad_dates), "order_date could not be read")

    # 3. Negative quantities = mis-encoded returns ---------------------------
    neg = df["quantity"] < 0
    df.loc[neg, "status"] = "returned"
    df["quantity"] = df["quantity"].abs()
    rep.log("recode negative quantity", int(neg.sum()), "flipped sign, marked returned")

    # 4. Fill missing status --------------------------------------------------
    n_miss_status = df["status"].isna().sum()
    df["status"] = df["status"].fillna("completed")
    rep.log("impute missing status", int(n_miss_status), "defaulted to 'completed'")

    # 5. Repair decimal-point typos in unit_price ----------------------------
    # A price more than 3x the product's list price is a data-entry error, not
    # a real transaction — divide by ten to undo the stray zero.
    df = df.merge(products[["product_id", "list_price", "category"]],
                  on="product_id", how="left")
    suspicious = df["unit_price"] > df["list_price"] * 3
    df.loc[suspicious, "unit_price"] = (df.loc[suspicious, "unit_price"] / 10).round(2)
    rep.log("repair price typos", int(suspicious.sum()), "value > 3x list price / 10")

    # 6. Recover missing unit_cost from the product master -------------------
    cost_lookup = products.set_index("product_id")["unit_cost"]
    miss_cost = df["unit_cost"].isna()
    df.loc[miss_cost, "unit_cost"] = df.loc[miss_cost, "product_id"].map(cost_lookup)
    rep.log("recover unit_cost", int(miss_cost.sum()), "joined from product master")

    # 7. Rebuild missing discount_pct from price vs list price ---------------
    miss_disc = df["discount_pct"].isna()
    df.loc[miss_disc, "discount_pct"] = (
        (1 - df.loc[miss_disc, "unit_price"] / df.loc[miss_disc, "list_price"]) * 100
    ).clip(lower=0).round(1)
    rep.log("recompute discount_pct", int(miss_disc.sum()), "derived from unit_price")

    # 8. Range validation -----------------------------------------------------
    valid = (
        df["unit_price"].between(C.MIN_UNIT_PRICE, C.MAX_UNIT_PRICE)
        & df["quantity"].between(C.MIN_QUANTITY, C.MAX_QUANTITY)
        & df["unit_cost"].notna()
    )
    rep.log("drop out-of-range rows", int((~valid).sum()), "price/quantity outside bounds")
    df = df[valid].copy()

    # 9. Derived measures -----------------------------------------------------
    df["revenue"] = (df["unit_price"] * df["quantity"]).round(2)
    df["cost"] = (df["unit_cost"] * df["quantity"]).round(2)
    df["profit"] = (df["revenue"] - df["cost"]).round(2)
    df["margin_pct"] = np.where(df["revenue"] > 0,
                                df["profit"] / df["revenue"] * 100, np.nan)

    # Returned lines earn no revenue but still cost money to process.
    returned = df["status"].eq("returned")
    df["net_revenue"] = np.where(returned, 0.0, df["revenue"])
    df["net_profit"] = np.where(returned, -df["cost"] * 0.15, df["profit"]).round(2)

    # 10. Flag (do not delete) revenue outliers ------------------------------
    q1, q3 = df["revenue"].quantile([0.25, 0.75])
    fence = q3 + C.IQR_MULTIPLIER * (q3 - q1)
    df["is_outlier"] = df["revenue"] > fence
    rep.log("flag revenue outliers", int(df["is_outlier"].sum()),
            f"revenue > {fence:,.0f} {C.CURRENCY} (IQR fence)")

    # 11. Calendar helpers ----------------------------------------------------
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["order_week"] = df["order_date"].dt.to_period("W").dt.start_time
    df["weekday"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek >= 5
    df["on_promo"] = df["discount_pct"] >= 20

    rep.rows_out = len(df)
    return df


def attach_customers(orders: pd.DataFrame, customers: pd.DataFrame,
                     rep: QualityReport) -> pd.DataFrame:
    c = customers.copy()
    c["country"] = normalise_country(c["country"])
    rep.log("normalise country labels", int(c["country"].nunique()),
            "trimmed, title-cased, codes expanded")

    c["signup_date"] = pd.to_datetime(c["signup_date"], errors="coerce")

    # Impute missing age with the country median — a weak but defensible rule.
    miss_age = c["age"].isna()
    c["age"] = c["age"].fillna(c.groupby("country")["age"].transform("median"))
    c["age"] = c["age"].fillna(c["age"].median()).round().astype(int)
    rep.log("impute missing age", int(miss_age.sum()), "country median")

    merged = orders.merge(c, on="customer_id", how="left", validate="many_to_one")
    orphans = merged["signup_date"].isna().sum()
    rep.log("orders with no customer", int(orphans), "left unmatched")
    return merged


def build_dataset(verbose: bool = True) -> tuple[pd.DataFrame, QualityReport]:
    orders_raw, customers_raw, products = load_raw()
    rep = QualityReport()
    df = clean_orders(orders_raw, products, rep)
    df = attach_customers(df, customers_raw, rep)

    if verbose:
        print(rep.render())
    return df, rep


def main() -> None:
    df, rep = build_dataset()
    path = save_table(df, C.CLEAN_ORDERS)
    rep.to_frame().to_csv(C.PROCESSED_DIR / "quality_report.csv", index=False)
    print(f"\nSaved {len(df):,} clean rows -> {path.name}")


if __name__ == "__main__":
    main()
