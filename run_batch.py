"""
run_batch.py -- the overnight sweep. Run this OUTSIDE Streamlit.

    python run_batch.py --replicates 10

3 samples x 9 tiles x 10 replicates = 270 DREAM.3D runs. Expect 1-4 hours.
It is RESUMABLE: Ctrl-C, close the laptop, rerun the same command -- completed
runs are read back from the CSV and skipped. Nothing is lost.

Then:
    python run_batch.py --stats-only        # tests on whatever is finished

EDIT THE CONFIG BLOCK BELOW before the first run:
  * ROI  -- must exclude the SEM databar, or the databar's black pixels are
            counted as pore and every volume fraction is wrong.
  * PERFORMANCE -- the measured ASR (or whatever figure of merit Dr. Jin
            provides) per sample. Leave as None until you have it; H2 is then
            skipped and H1 still runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from psm.batch import SampleSpec, BatchConfig, run_batch
from psm import stats as pstats

# ===========================================================================
# CONFIG -- edit this
# ===========================================================================
TEMPLATE = "3phasemicrostructure_stl_removeisolated (2).json"
RUNNER = r"C:\DREAM3D\DREAM3D-6.5.171-Win64\PipelineRunner.exe"
OUT_DIR = "outputs/batch"

UM_PER_PX = 0.0365          # validated click-to-measure calibration

# roi = (row0, col0, height, width). MUST crop out the databar.
# The OFA images are 2048x1887 native; the databar is the bottom strip.
# Verify the number by eye in Stage 1 before trusting this.
ROI = (0, 0, 1887, 2048)

# Per-sample bands, validated against the MATLAB reference. Applied UNCHANGED
# to every tile of that sample -- do NOT re-threshold per tile.
SAMPLES = [
    SampleSpec("OFA8",  "images/OFA8.tif",  UM_PER_PX,
               [("Pore", 0, 100), ("NiO", 101, 174), ("BZCYYb", 175, 255)],
               (564, 0, 1202, 2048)),
    SampleSpec("OFA10", "images/OFA10.tif", UM_PER_PX,
               [("Pore", 0, 75),  ("NiO", 76, 180),  ("BZCYYb", 181, 255)],
               (523, 0, 1243, 2048)),
    SampleSpec("OFA11", "images/OFA11.tif", UM_PER_PX,
               [("Pore", 0, 59),  ("NiO", 60, 190),  ("BZCYYb", 191, 255)],
               (734, 0, 1032, 2048)),
]

# Measured electrochemical performance. Fill in from Dr. Jin.
# Lower ASR = better electrode -> HIGHER_PERF_IS_BETTER = False.
PERFORMANCE = None      # e.g. {"OFA8": 0.42, "OFA10": 0.31, "OFA11": 0.55}
HIGHER_PERF_IS_BETTER = False

# Headline metrics for the paper. The full sweep is tested anyway; these are
# the ones printed to the console.
HEADLINE = ["tau_z_Pore", "tau_z_NiO", "tau_z_BZCYYb",
            "tpb_density_um-2", "conn_Pore", "conn_NiO", "conn_BZCYYb",
            "vf3d_Pore"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=10,
                    help="DREAM.3D runs per tile (reconstruction noise sample)")
    ap.add_argument("--grid", type=int, default=3, help="GRID x GRID tiles")
    ap.add_argument("--stats-only", action="store_true",
                    help="skip the sweep; just re-run the tests on the CSV")
    args = ap.parse_args()

    out = Path(OUT_DIR)
    csv = out / "metrics_long.csv"

    if not args.stats_only:
        cfg = BatchConfig(template=TEMPLATE, runner_exe=RUNNER, out_dir=OUT_DIR,
                          grid=args.grid, replicates=args.replicates)
        total = len(SAMPLES) * args.grid ** 2 * args.replicates
        print(f"sweep: {len(SAMPLES)} samples x {args.grid**2} tiles "
              f"x {args.replicates} reps = {total} DREAM.3D runs\n")

        def prog(done, tot, msg):
            bar = "#" * int(30 * done / tot)
            print(f"\r[{bar:<30}] {done}/{tot}  {msg[:44]:<44}", end="", flush=True)

        run_batch(SAMPLES, cfg, progress=prog)
        print("\n")

    if not csv.exists():
        print(f"no results at {csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    fmt = lambda x: f"{x:.4g}"

    print("=" * 78)
    print("EXCLUSIONS  (report this table -- it is part of the method)")
    print("=" * 78)
    print(pstats.excluded_report(df).to_string(index=False))

    print("\n" + "=" * 78)
    print("H1 -- do the three electrodes differ?   Kruskal-Wallis")
    print("=" * 78)
    h1 = pstats.h1_table(df, metrics=HEADLINE)
    if not h1.empty:
        cols = [c for c in h1.columns if not c.startswith("n_") or c.startswith("n_inf")]
        print(h1[cols].to_string(index=False, float_format=fmt))
        h1.to_csv(out / "h1_kruskal.csv", index=False)

        print("\n--- Dunn post-hoc (Holm-corrected) on significant metrics ---")
        d = pstats.clean(df)
        posts = [pstats.dunn_posthoc(d, m)
                 for m in h1[h1["significant_05"]]["metric"]]
        if posts:
            allp = pd.concat(posts, ignore_index=True)
            print(allp.to_string(index=False, float_format=fmt))
            allp.to_csv(out / "h1_dunn.csv", index=False)

    print("\n" + "=" * 78)
    print("VARIANCE -- electrode heterogeneity vs DREAM.3D packing noise")
    print("noise_share near 1.0 means the metric is measuring the ALGORITHM.")
    print("=" * 78)
    vc = pstats.variance_components(df, metrics=HEADLINE)
    if not vc.empty:
        print(vc[["sample", "metric", "tile_var", "replicate_var", "noise_share"]]
              .to_string(index=False, float_format=fmt))
        vc.to_csv(out / "variance_components.csv", index=False)
        hi = vc[vc["noise_share"] > 0.5]["metric"].unique()
        if len(hi):
            print(f"\n  !! noise-dominated (>50%): {', '.join(hi)}")
            print("     Raise --replicates, or do not report these as material "
                  "properties without saying so.")

    print("\n" + "=" * 78)
    print("H2 -- does the metric ranking match performance?")
    print("=" * 78)
    if PERFORMANCE is None:
        print("  PERFORMANCE is None -- fill it in from Dr. Jin's data. H2 skipped.")
    else:
        h2 = pstats.spearman_vs_performance(
            df, PERFORMANCE, metrics=HEADLINE,
            higher_perf_is_better=HIGHER_PERF_IS_BETTER)
        print(h2[["metric", "spearman_rho", "metric_rank_order",
                  "perf_rank_order", "ranks_match"]].to_string(index=False))
        h2.to_csv(out / "h2_spearman.csv", index=False)
        print("\n  n=3: this is a RANKING statement, not a correlation. "
              "No p-value. No regression.")

    print(f"\nwrote CSVs to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
