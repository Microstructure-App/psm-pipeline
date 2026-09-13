"""
psm.stats -- the statistical tests for H1 and H2.

Consumes the long CSV written by ``psm.batch.run_batch`` (one row per
sample x tile x replicate) and produces the numbers that go in the paper.

H1 -- RESOLUTION.  Do OFA8, OFA10 and OFA11 differ in transport metrics by
more than within-field heterogeneity and reconstruction noise?
    Test: Kruskal-Wallis H across the three samples, per metric.
    Post-hoc: Dunn's test with Holm correction on the pairs.
    Effect size: epsilon-squared.
Kruskal-Wallis is used rather than one-way ANOVA because (a) nothing
guarantees the per-tile metric distributions are normal, (b) n is small, and
(c) it is rank-based, so a non-percolating tau = INF ranks correctly as
"worst" instead of poisoning the test with NaN or an arbitrary large number.
That last property is the reason inf is the right sentinel and NaN is not.

VARIANCE DECOMPOSITION.  The Kruskal-Wallis alone does not say WHERE the
spread comes from. ``variance_components`` splits the observed variance into
the tile-to-tile part (real spatial heterogeneity of the electrode) and the
replicate-to-replicate part (DREAM.3D packing noise). If reconstruction noise
dominates tile heterogeneity, the metric is not measuring the material and
should not be reported as if it were -- report this table alongside the test.

H2 -- RELEVANCE.  Does the metric ranking match the performance ranking?
    ``spearman_vs_performance`` returns rho for n = 3 samples and DELIBERATELY
    returns no p-value. With three points a p-value is meaningless (the
    smallest attainable two-sided p for a perfect monotone rank relation at
    n = 3 is 0.333) and reporting one would be misleading. Report rho as a
    description of ordering, plus the ranking table itself, and say in the
    text that n = 3 supports ranking only, not correlation.

NOTHING IN THIS MODULE FITS A REGRESSION THROUGH THREE POINTS. Do not add one.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


# ===========================================================================
# Helpers
# ===========================================================================
def metric_columns(df: pd.DataFrame) -> List[str]:
    """Every numeric metric column worth testing."""
    keep = []
    for c in df.columns:
        if c.startswith(("tau_", "conn_", "vf3d_", "vf2d_", "tpb_", "dpb_",
                         "mu_", "sigma_", "meand_", "nfeat_")):
            keep.append(c)
    return keep


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop failed runs and runs whose lognormal fit collapsed.

    A row with psd_ok == False had at least one phase fall back to DREAM.3D's
    TEMPLATE DEFAULT (mu=0.7, sigma=0.2) instead of its measured distribution.
    That row describes the template, not the electrode. It must never enter a
    test -- but the COUNT of such rows must be reported, because a sample that
    loses many tiles this way (pore is the usual culprit: too few particles)
    is telling you something real about the segmentation.
    """
    d = df.copy()
    if "status" in d:
        d = d[d["status"] == "ok"]
    if "psd_ok" in d:
        d = d[d["psd_ok"].astype(str).str.lower().isin(["true", "1"])]
    return d.reset_index(drop=True)


def excluded_report(df: pd.DataFrame) -> pd.DataFrame:
    """How many runs were dropped, per sample, and why. Goes in the paper."""
    d = df.copy()
    rows = []
    for s, g in d.groupby("sample"):
        failed = int((g.get("status", pd.Series(["ok"] * len(g))) != "ok").sum())
        badpsd = 0
        if "psd_ok" in g:
            badpsd = int((~g["psd_ok"].astype(str).str.lower()
                          .isin(["true", "1"])).sum())
        bad_names = ""
        if "bad_phases" in g:
            names = [x for x in g["bad_phases"].dropna().astype(str) if x]
            bad_names = ";".join(sorted({p for x in names for p in x.split(";") if p}))
        rows.append(dict(sample=s, n_total=len(g), n_dream3d_failed=failed,
                         n_psd_fit_failed=badpsd,
                         n_used=len(g) - failed - badpsd,
                         phases_that_failed_fit=bad_names))
    return pd.DataFrame(rows)


# ===========================================================================
# H1 -- Kruskal-Wallis + Dunn
# ===========================================================================
@dataclass
class KWResult:
    metric: str
    H: float
    p: float
    epsilon_sq: float          # effect size, 0..1
    n_per_group: Dict[str, int]
    median_per_group: Dict[str, float]
    n_inf_per_group: Dict[str, int]     # non-percolating runs (tau only)
    note: str = ""


def kruskal_by_sample(df: pd.DataFrame, metric: str,
                      group_col: str = "sample") -> Optional[KWResult]:
    """Kruskal-Wallis across samples for one metric. None if not testable."""
    groups, meds, ns, ninf = {}, {}, {}, {}
    for s, g in df.groupby(group_col):
        v = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        groups[s] = v
        ns[s] = int(v.size)
        ninf[s] = int(np.isinf(v).sum())
        meds[s] = float(np.median(v))    # median tolerates inf if <50% are inf

    if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
        return None

    # Constant across every group -> H is undefined and the question is moot.
    allv = np.concatenate(list(groups.values()))
    if np.all(np.isinf(allv)):
        return KWResult(metric, float("nan"), float("nan"), float("nan"),
                        ns, meds, ninf,
                        note="phase never percolates in any run; test not applicable")
    if len(np.unique(allv[np.isfinite(allv)])) <= 1 and not np.isinf(allv).any():
        return KWResult(metric, float("nan"), float("nan"), float("nan"),
                        ns, meds, ninf, note="metric is constant; no variance to test")

    H, p = sps.kruskal(*groups.values())
    N = int(sum(ns.values()))
    k = len(groups)
    # epsilon^2 = (H - k + 1) / (N - k)   [Tomczak & Tomczak 2014]
    eps = (H - k + 1) / (N - k) if N > k else float("nan")
    eps = float(max(0.0, min(1.0, eps)))

    note = ""
    tot_inf = sum(ninf.values())
    if tot_inf:
        note = (f"{tot_inf}/{N} runs non-percolating (tau=inf); ranks handle "
                "these correctly but report the count explicitly")
    return KWResult(metric, float(H), float(p), eps, ns, meds, ninf, note)


def dunn_posthoc(df: pd.DataFrame, metric: str,
                 group_col: str = "sample") -> pd.DataFrame:
    """Dunn's test on all sample pairs, Holm-corrected.

    Implemented directly (no scikit-posthocs dependency) so the package stays
    installable from requirements.txt alone. Ranks are computed on the POOLED
    sample, which is what makes this the correct post-hoc for Kruskal-Wallis
    rather than a series of pairwise Mann-Whitney tests.
    """
    d = df[[group_col, metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    if d.empty:
        return pd.DataFrame()

    # rankdata handles +inf: it is simply the largest value.
    d["rank"] = sps.rankdata(d[metric].to_numpy(dtype=float))
    N = len(d)
    g = d.groupby(group_col)["rank"]
    means, ns = g.mean().to_dict(), g.size().to_dict()

    # tie correction
    _, counts = np.unique(d[metric].to_numpy(dtype=float), return_counts=True)
    ties = float(np.sum(counts ** 3 - counts))
    sigma_base = (N * (N + 1) / 12.0) - (ties / (12.0 * (N - 1))) if N > 1 else 0.0

    rows = []
    for a, b in itertools.combinations(sorted(means), 2):
        se = math.sqrt(sigma_base * (1.0 / ns[a] + 1.0 / ns[b])) if sigma_base > 0 else 0.0
        z = (means[a] - means[b]) / se if se > 0 else 0.0
        p = 2.0 * (1.0 - sps.norm.cdf(abs(z)))
        rows.append(dict(metric=metric, group_a=a, group_b=b,
                         z=float(z), p_raw=float(p)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm step-down
    out = out.sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    adj, running = [], 0.0
    for i, p in enumerate(out["p_raw"]):
        running = max(running, (m - i) * p)
        adj.append(min(1.0, running))
    out["p_holm"] = adj
    out["significant_05"] = out["p_holm"] < 0.05
    return out


def h1_table(df: pd.DataFrame, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The H1 results table: one row per metric."""
    d = clean(df)
    mets = list(metrics) if metrics else metric_columns(d)
    rows = []
    for m in mets:
        r = kruskal_by_sample(d, m)
        if r is None:
            continue
        row = dict(metric=m, H=r.H, p=r.p, epsilon_sq=r.epsilon_sq,
                   significant_05=(r.p < 0.05) if math.isfinite(r.p) else False,
                   note=r.note)
        for s in sorted(r.median_per_group):
            row[f"median_{s}"] = r.median_per_group[s]
            row[f"n_{s}"] = r.n_per_group[s]
            if r.n_inf_per_group.get(s):
                row[f"n_inf_{s}"] = r.n_inf_per_group[s]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("p", na_position="last").reset_index(drop=True) if not out.empty else out


# ===========================================================================
# Variance decomposition -- where does the spread actually come from?
# ===========================================================================
def variance_components(df: pd.DataFrame,
                        metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Split within-sample variance into tile (real) and replicate (noise) parts.

    For each sample: replicate variance is the mean of the per-tile variances
    (spread with the input held fixed -- pure DREAM.3D packing noise). Tile
    variance is the variance of the per-tile means, minus the replicate
    variance divided by the number of replicates (the standard one-way
    random-effects correction; without it, tile variance is inflated by the
    noise that leaks into each tile mean).

    ``noise_share`` = replicate / (tile + replicate). Read it as: what fraction
    of the wobble is the ALGORITHM rather than the ELECTRODE. A metric with a
    high noise share is not measuring the material.

    Rows with any inf are dropped for this table only -- variance is undefined
    with inf, unlike the rank test. The dropped count is reported.
    """
    d = clean(df)
    mets = list(metrics) if metrics else metric_columns(d)
    rows = []
    for s, g in d.groupby("sample"):
        for m in mets:
            v = pd.to_numeric(g[m], errors="coerce")
            ok = np.isfinite(v)
            n_dropped = int((~ok).sum())
            gg = g[ok.to_numpy()]
            if gg["tile"].nunique() < 2:
                continue
            per_tile = gg.groupby("tile")[m].agg(["mean", "var", "count"])
            # Replicate variance needs >=2 replicates per tile. With --replicates 1
            # per_tile["var"] is all-NaN, and silently coercing that to 0 would
            # report "zero reconstruction noise", which is a lie -- the noise was
            # never measured. Skip the metric and say so.
            if float(per_tile["count"].max()) < 2:
                rows.append(dict(
                    sample=s, metric=m, tile_var=float("nan"),
                    replicate_var=float("nan"), noise_share=float("nan"),
                    n_tiles=int(gg["tile"].nunique()),
                    mean_reps_per_tile=float(per_tile["count"].mean()),
                    n_dropped_nonfinite=n_dropped,
                    note="needs >=2 replicates per tile to separate noise from heterogeneity"))
                continue
            rep_var = float(np.nanmean(per_tile["var"]))
            n_rep = float(np.nanmean(per_tile["count"]))
            tile_var = float(np.var(per_tile["mean"], ddof=1))
            tile_var_corr = max(0.0, tile_var - (rep_var / n_rep if n_rep else 0.0))
            tot = tile_var_corr + rep_var
            rows.append(dict(
                sample=s, metric=m,
                tile_var=tile_var_corr, replicate_var=rep_var,
                noise_share=(rep_var / tot) if tot > 0 else float("nan"),
                n_tiles=int(gg["tile"].nunique()),
                mean_reps_per_tile=n_rep,
                n_dropped_nonfinite=n_dropped, note=""))
    return pd.DataFrame(rows)


# ===========================================================================
# H2 -- ranking against performance (n = 3; NO p-value, NO regression)
# ===========================================================================
def spearman_vs_performance(df: pd.DataFrame,
                            performance: Dict[str, float],
                            metrics: Optional[Sequence[str]] = None,
                            higher_perf_is_better: bool = False) -> pd.DataFrame:
    """Rank-consistency of each metric with measured performance, n = 3 samples.

    ``performance`` maps sample id -> the measured figure of merit (e.g. ASR in
    ohm cm^2). Set ``higher_perf_is_better=False`` for ASR / polarization
    resistance (lower = better electrode); True for peak power density.

    Returns rho and the two rank orders. NO P-VALUE IS RETURNED, BY DESIGN --
    see the module docstring. The column ``ranks_match`` is the honest headline:
    does the metric order the three electrodes the same way performance does?
    """
    d = clean(df)
    mets = list(metrics) if metrics else metric_columns(d)
    samples = [s for s in sorted(d["sample"].unique()) if s in performance]
    if len(samples) < 3:
        raise ValueError(
            f"need performance data for >=3 samples; have {len(samples)}: {samples}")

    perf = np.array([performance[s] for s in samples], dtype=float)
    # Orient so that a HIGHER perf_score always means a BETTER electrode.
    perf_score = perf if higher_perf_is_better else -perf

    rows = []
    for m in mets:
        med = []
        for s in samples:
            v = pd.to_numeric(d[d["sample"] == s][m], errors="coerce").to_numpy(float)
            v = v[~np.isnan(v)]
            med.append(float(np.median(v)) if v.size else float("nan"))
        med = np.array(med, dtype=float)
        if not np.all(np.isfinite(med)):
            # A metric that is inf (non-percolating) for some sample cannot be
            # rank-correlated -- but that is itself the finding. Report it.
            rows.append(dict(
                metric=m,
                **{f"median_{s}": med[i] for i, s in enumerate(samples)},
                spearman_rho=float("nan"),
                metric_rank_order="not rankable (inf / undefined)",
                perf_rank_order=" < ".join(np.array(samples)[np.argsort(perf)]),
                ranks_match=False, n_samples=len(samples),
                caveat="metric is inf or undefined for >=1 sample; report the "
                       "non-percolation directly, do not rank"))
            continue
        if len(np.unique(med)) < len(med):
            # Ties make the rank order arbitrary; scipy returns nan rho for a
            # fully constant vector and silently breaks ties otherwise. Neither
            # is reportable at n=3.
            rows.append(dict(
                metric=m,
                **{f"median_{s}": med[i] for i, s in enumerate(samples)},
                spearman_rho=float("nan"),
                metric_rank_order="tied (no unique order)",
                perf_rank_order=" < ".join(np.array(samples)[np.argsort(perf)]),
                ranks_match=False, n_samples=len(samples),
                caveat="metric medians are tied across samples; it does not "
                       "discriminate these electrodes"))
            continue
        rho = float(sps.spearmanr(med, perf_score).statistic)
        rows.append(dict(
            metric=m,
            **{f"median_{s}": med[i] for i, s in enumerate(samples)},
            spearman_rho=rho,
            metric_rank_order=" < ".join(np.array(samples)[np.argsort(med)]),
            perf_rank_order=" < ".join(np.array(samples)[np.argsort(perf)]),
            ranks_match=bool(abs(rho) == 1.0),
            n_samples=len(samples),
            caveat="n=3: ordering only, not correlation; no p-value is meaningful"))
    out = pd.DataFrame(rows)
    return (out.reindex(out["spearman_rho"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True)) if not out.empty else out