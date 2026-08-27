"""Generate a synthetic — but realistic — retail dataset.

The generator deliberately injects the kinds of defects you meet in real
exports (duplicated rows, mixed date formats, decimal-point typos, inconsistent
country spellings, missing values) so the cleaning stage in `data_prep.py` has
something genuine to do.

Run:  python -m src.generate_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

rng = np.random.default_rng(C.RANDOM_SEED)

CATEGORIES = {
    # category: (n_products, price_lognormal_mean, price_sigma, margin)
    "Electronics":    (24, 5.2, 0.55, 0.22),
    "Home & Kitchen": (22, 3.9, 0.50, 0.38),
    "Apparel":        (26, 3.5, 0.45, 0.52),
    "Sports":         (18, 4.1, 0.55, 0.35),
    "Beauty":         (18, 2.9, 0.40, 0.60),
    "Toys":           (12, 3.2, 0.50, 0.44),
}

COUNTRIES = ["Germany", "France", "Netherlands", "Austria", "Spain", "Italy"]
COUNTRY_P = [0.42, 0.16, 0.13, 0.11, 0.10, 0.08]

CHANNELS = ["Organic Search", "Paid Social", "Email", "Referral", "Direct"]
CHANNEL_P = [0.31, 0.24, 0.18, 0.12, 0.15]


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def make_products() -> pd.DataFrame:
    rows = []
    pid = 1000
    for cat, (n, mu, sigma, margin) in CATEGORIES.items():
        for _ in range(n):
            pid += 1
            price = float(np.exp(rng.normal(mu, sigma)))
            price = round(min(max(price, 3.0), 2_400.0), 2)
            rows.append(
                {
                    "product_id": f"P{pid}",
                    "product_name": f"{cat.split()[0]} Item {pid}",
                    "category": cat,
                    "unit_cost": round(price * (1 - margin), 2),
                    "list_price": price,
                }
            )
    df = pd.DataFrame(rows)
    # Popularity follows a long tail: a few products drive most volume. The
    # shape parameter is tuned so the top SKU takes a realistic single-digit
    # share of units — a heavier tail lets one product swamp the whole catalogue.
    pop = rng.pareto(2.2, len(df)) + 1
    df["popularity"] = pop / pop.sum()
    return df


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def make_customers() -> pd.DataFrame:
    n = C.N_CUSTOMERS

    # The business did not start on day one of the analysis window: roughly
    # half the base was acquired in the two years before it. Without this the
    # first months contain almost no customers and every trend looks explosive.
    history_start = pd.Timestamp(C.START_DATE) - pd.DateOffset(years=2)
    all_days = pd.date_range(history_start, C.END_DATE, freq="D")

    # Acquisition grows steadily over time, so signups skew toward later dates.
    weights = np.linspace(0.5, 1.8, len(all_days))
    weights = weights / weights.sum()
    signup_idx = rng.choice(len(all_days), size=n, p=weights)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{50000 + i}" for i in range(n)],
            "signup_date": all_days[signup_idx],
            "country": rng.choice(COUNTRIES, size=n, p=COUNTRY_P),
            "acquisition_channel": rng.choice(CHANNELS, size=n, p=CHANNEL_P),
            "age": np.clip(rng.normal(38, 12, n), 18, 85).round().astype(int),
        }
    )
    # Latent per-customer purchase propensity — drives repeat behaviour.
    df["propensity"] = rng.gamma(shape=1.6, scale=1.0, size=n)
    return df


# ---------------------------------------------------------------------------
# Daily demand weights: trend + yearly seasonality + weekday + promo spikes
# ---------------------------------------------------------------------------
def daily_weights(days: pd.DatetimeIndex) -> np.ndarray:
    t = np.arange(len(days))
    trend = 1.0 + 0.55 * t / len(days)

    doy = days.dayofyear.to_numpy()
    seasonal = 1.0 + 0.22 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    weekday = np.where(days.dayofweek.to_numpy() >= 5, 1.25, 1.0)

    promo = np.ones(len(days))
    for i, d in enumerate(days):
        if d.month == 11 and 22 <= d.day <= 30:      # Black Friday window
            promo[i] = 3.4
        elif d.month == 12 and d.day <= 20:          # Christmas run-up
            promo[i] = 2.1
        elif d.month == 1 and d.day <= 15:           # January sales
            promo[i] = 1.5
        elif d.month in (7, 8):                      # summer lull
            promo[i] = 0.85

    w = trend * seasonal * weekday * promo
    return w / w.sum()


def is_promo_day(days: pd.DatetimeIndex) -> np.ndarray:
    m, d = days.month.to_numpy(), days.day.to_numpy()
    return (
        ((m == 11) & (d >= 22) & (d <= 30))
        | ((m == 12) & (d <= 20))
        | ((m == 1) & (d <= 15))
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
def make_orders(customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(C.START_DATE, C.END_DATE, freq="D")
    w = daily_weights(days)
    cum_w = np.cumsum(w)

    signup_pos = np.searchsorted(days.to_numpy(), customers["signup_date"].to_numpy())

    # Orders per customer: heavier tail for high-propensity customers.
    lam = 0.9 * customers["propensity"].to_numpy()
    n_orders = rng.poisson(lam)
    n_orders = np.clip(n_orders, 0, 18)

    cust_idx = np.repeat(np.arange(len(customers)), n_orders)
    if len(cust_idx) == 0:
        raise RuntimeError("No orders generated — check propensity settings.")

    # Draw each order date from the daily distribution, truncated to be on or
    # after that customer's signup date.
    lo = np.where(signup_pos[cust_idx] > 0, cum_w[signup_pos[cust_idx] - 1], 0.0)
    hi = cum_w[-1]
    u = rng.uniform(lo, hi)
    day_pos = np.searchsorted(cum_w, u)
    day_pos = np.clip(day_pos, signup_pos[cust_idx], len(days) - 1)
    order_dates = days[day_pos]

    n_orders_total = len(cust_idx)
    order_ids = np.array([f"O{200000 + i}" for i in range(n_orders_total)])

    # Basket size 1-6, skewed small.
    basket = rng.choice([1, 2, 3, 4, 5, 6], size=n_orders_total,
                        p=[0.34, 0.27, 0.18, 0.11, 0.06, 0.04])

    line_order = np.repeat(np.arange(n_orders_total), basket)
    n_lines = len(line_order)

    prod_pos = rng.choice(len(products), size=n_lines, p=products["popularity"].to_numpy())
    prod = products.iloc[prod_pos]

    quantity = rng.choice([1, 2, 3, 4, 5], size=n_lines,
                          p=[0.62, 0.21, 0.10, 0.05, 0.02])

    # Discounts: deeper and more frequent on promo days.
    line_dates = order_dates[line_order]
    promo_flag = is_promo_day(line_dates)
    base_disc = rng.choice([0.0, 0.05, 0.10, 0.15], size=n_lines,
                           p=[0.72, 0.13, 0.10, 0.05])
    promo_disc = rng.choice([0.10, 0.20, 0.30, 0.40], size=n_lines,
                            p=[0.25, 0.35, 0.28, 0.12])
    discount = np.where(promo_flag, promo_disc, base_disc)

    list_price = prod["list_price"].to_numpy()
    unit_price = np.round(list_price * (1 - discount), 2)

    df = pd.DataFrame(
        {
            "order_id": order_ids[line_order],
            "order_line": np.concatenate([np.arange(1, b + 1) for b in basket]),
            "order_date": line_dates,
            "customer_id": customers["customer_id"].to_numpy()[cust_idx][line_order],
            "product_id": prod["product_id"].to_numpy(),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_pct": np.round(discount * 100, 1),
            "unit_cost": prod["unit_cost"].to_numpy(),
        }
    )

    # Returns: ~4% of lines come back, encoded as a status column.
    df["status"] = np.where(rng.random(n_lines) < 0.04, "returned", "completed")
    return df.sort_values(["order_date", "order_id", "order_line"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Corruption: make the raw files look like a real export
# ---------------------------------------------------------------------------
def corrupt(orders: pd.DataFrame, customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = orders.copy()
    c = customers.copy()

    # 1. Mixed date formats stored as text.
    dates = o["order_date"]
    fmt = rng.random(len(o))
    as_text = np.where(
        fmt < 0.75,
        dates.dt.strftime("%Y-%m-%d"),
        np.where(fmt < 0.92,
                 dates.dt.strftime("%d/%m/%Y"),
                 dates.dt.strftime("%b %d, %Y")),
    )
    o["order_date"] = as_text

    # 2. Decimal-point typos — an extra zero on some prices.
    typo = rng.random(len(o)) < 0.004
    o.loc[typo, "unit_price"] = o.loc[typo, "unit_price"] * 10

    # 3. Negative quantities used (incorrectly) to mark returns.
    neg = (o["status"] == "returned") & (rng.random(len(o)) < 0.35)
    o.loc[neg, "quantity"] = -o.loc[neg, "quantity"]

    # 4. Missing values scattered through non-key columns.
    for col, frac in [("discount_pct", 0.03), ("unit_cost", 0.02), ("status", 0.01)]:
        miss = rng.random(len(o)) < frac
        o.loc[miss, col] = np.nan

    # 5. Exact duplicate rows (double-submitted batch).
    dupes = o.sample(frac=0.015, random_state=C.RANDOM_SEED)
    o = pd.concat([o, dupes], ignore_index=True)

    # 6. Whitespace and casing noise in categoricals.
    noisy = rng.random(len(c)) < 0.08
    c.loc[noisy, "country"] = c.loc[noisy, "country"].str.upper()
    noisy2 = rng.random(len(c)) < 0.05
    c.loc[noisy2, "country"] = "  " + c.loc[noisy2, "country"] + " "
    c.loc[c.sample(frac=0.02, random_state=1).index, "country"] = "DE"

    # 7. Missing ages.
    c.loc[c.sample(frac=0.06, random_state=2).index, "age"] = np.nan

    # Shuffle so defects are not clustered at the end of the file.
    o = o.sample(frac=1.0, random_state=C.RANDOM_SEED).reset_index(drop=True)
    return o, c


def main() -> None:
    print("Generating synthetic retail dataset...")
    products = make_products()
    customers = make_customers()
    orders = make_orders(customers, products)
    orders_raw, customers_raw = corrupt(orders, customers)

    products.drop(columns=["popularity"]).to_csv(C.RAW_PRODUCTS, index=False)
    customers_raw.drop(columns=["propensity"]).to_csv(C.RAW_CUSTOMERS, index=False)
    orders_raw.to_csv(C.RAW_ORDERS, index=False)

    print(f"  products : {len(products):>7,} rows -> {C.RAW_PRODUCTS.name}")
    print(f"  customers: {len(customers_raw):>7,} rows -> {C.RAW_CUSTOMERS.name}")
    print(f"  orders   : {len(orders_raw):>7,} rows -> {C.RAW_ORDERS.name}")


if __name__ == "__main__":
    main()
