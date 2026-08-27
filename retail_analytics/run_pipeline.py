#!/usr/bin/env python3
"""Run the whole project end to end.

    python run_pipeline.py              # full run
    python run_pipeline.py --skip-data  # reuse existing raw data
    python run_pipeline.py --no-figures # numbers only, no plotting

Stages: generate -> clean -> analyse -> plot -> report.
"""

from __future__ import annotations

import argparse
import sys
import time

from src import analysis, config as C, data_prep, generate_data, report, viz
from src.io_utils import save_table


def _banner(step: int, total: int, title: str) -> float:
    print(f"\n[{step}/{total}] {title}")
    print("-" * 64)
    return time.perf_counter()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retail analytics pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="reuse the existing raw CSVs instead of regenerating")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip chart generation")
    parser.add_argument("--no-report", action="store_true",
                        help="skip the markdown report")
    args = parser.parse_args(argv)

    total = 5
    started = time.perf_counter()

    # 1. Generate ------------------------------------------------------------
    t = _banner(1, total, "Generate raw data")
    if args.skip_data and C.RAW_ORDERS.exists():
        print("  skipped — using existing raw files")
    else:
        generate_data.main()
    print(f"  {time.perf_counter() - t:.2f}s")

    # 2. Clean ---------------------------------------------------------------
    t = _banner(2, total, "Clean and validate")
    df, rep = data_prep.build_dataset(verbose=True)
    save_table(df, C.CLEAN_ORDERS)
    rep.to_frame().to_csv(C.PROCESSED_DIR / "quality_report.csv", index=False)
    print(f"  {time.perf_counter() - t:.2f}s")

    # 3. Analyse -------------------------------------------------------------
    t = _banner(3, total, "Analyse")
    res = analysis.run_all(df)
    k = res["kpis"]
    print(f"  net revenue     {k['net_revenue']:>14,.0f} {C.CURRENCY}")
    print(f"  net profit      {k['net_profit']:>14,.0f} {C.CURRENCY} "
          f"({k['margin_pct']:.1f}%)")
    print(f"  orders          {k['orders']:>14,}")
    print(f"  customers       {k['customers']:>14,}")
    print(f"  AOV             {k['aov']:>14,.2f} {C.CURRENCY}")
    print(f"  repeat rate     {k['repeat_rate_pct']:>13.1f}%")

    save_table(res["rfm"], C.RFM_TABLE)
    res["retention"].to_csv(C.COHORT_TABLE)
    save_table(res["category"], C.PROCESSED_DIR / "category_performance.csv")
    save_table(res["segments"], C.PROCESSED_DIR / "segment_profile.csv")
    print(f"  {time.perf_counter() - t:.2f}s")

    # 4. Plot ----------------------------------------------------------------
    t = _banner(4, total, "Build figures")
    if args.no_figures:
        print("  skipped")
    else:
        for p in viz.build_all(res):
            print(f"  {p.relative_to(C.PROJECT_ROOT)}")
    print(f"  {time.perf_counter() - t:.2f}s")

    # 5. Report --------------------------------------------------------------
    t = _banner(5, total, "Write report")
    if args.no_report:
        print("  skipped")
    else:
        text = report.build_report(res)
        path = C.REPORTS_DIR / "findings.md"
        path.write_text(text, encoding="utf-8")
        print(f"  {path.relative_to(C.PROJECT_ROOT)}")
    print(f"  {time.perf_counter() - t:.2f}s")

    print("\n" + "=" * 64)
    print(f"Pipeline finished in {time.perf_counter() - started:.2f}s")
    print(f"Report:  {(C.REPORTS_DIR / 'findings.md')}")
    print(f"Figures: {C.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
