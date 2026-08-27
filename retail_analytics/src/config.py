"""Central configuration for the retail analytics project.

Keeping paths and tunable constants in one module means no other file needs
to hardcode a directory or a magic number.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_ORDERS = RAW_DIR / "orders_raw.csv"
RAW_CUSTOMERS = RAW_DIR / "customers_raw.csv"
RAW_PRODUCTS = RAW_DIR / "products_raw.csv"

CLEAN_ORDERS = PROCESSED_DIR / "orders_clean.parquet"
RFM_TABLE = PROCESSED_DIR / "rfm_segments.csv"
COHORT_TABLE = PROCESSED_DIR / "cohort_retention.csv"

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Simulation settings ---------------------------------------------------
RANDOM_SEED = 42
N_CUSTOMERS = 4_000
N_PRODUCTS = 120
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# --- Data quality rules ----------------------------------------------------
# An order line outside these bounds is treated as a data-entry error.
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 5_000.00
MIN_QUANTITY = 1
MAX_QUANTITY = 100

# Revenue outliers are flagged, not deleted, using the IQR fence multiplier.
IQR_MULTIPLIER = 3.0

# --- Analysis settings -----------------------------------------------------
RFM_QUANTILES = 5           # score each RFM dimension 1-5
COHORT_MAX_PERIODS = 12     # months of retention to track
CURRENCY = "EUR"

# --- Plotting --------------------------------------------------------------
FIG_DPI = 150
FIG_SIZE_WIDE = (11, 5.5)
FIG_SIZE_SQUARE = (8, 7)

PALETTE = {
    "primary": "#2B6CB0",
    "secondary": "#DD6B20",
    "accent": "#38A169",
    "muted": "#A0AEC0",
    "dark": "#1A202C",
    "danger": "#C53030",
}

CATEGORY_COLORS = {
    "Electronics": "#2B6CB0",
    "Home & Kitchen": "#DD6B20",
    "Apparel": "#38A169",
    "Sports": "#805AD5",
    "Beauty": "#D53F8C",
    "Toys": "#D69E2E",
}
