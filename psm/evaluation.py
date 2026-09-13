"""
Metric reporting, replicate statistics, and cross-sample comparison.

DESIGN NOTE — why there is no scorecard here anymore
----------------------------------------------------
An earlier version of this module judged each metric against literature
thresholds and emitted a ✅/🟡/❌ verdict. That was removed deliberately:

  1. The thresholds were drawn from the Ni-YSZ SOFC anode literature and were
     being applied to arbitrary user material. There is no defensible reason
     for this tool to assert that (say) 2.0 µm⁻² is a "good" TPB density for
     a structure it knows nothing about.
  2. Most of the judged metrics are computed on a DREAM.3D-*synthesized*
     volume built from 2D statistics. Attaching a quality verdict to a
     synthetic structure quietly promotes it to a measurement of the real one.
  3. A phase that does not percolate is a *finding*, not a defect. Marking it
     ❌ pre-judges a result the user may specifically be looking for (e.g. an
     as-sintered, pre-reduction electrode is not expected to percolate).
  4. The τ / TPB "calibration factors" silently multiplied reported values.
     A fudge factor applied before reporting is indefensible in a publication
     and has been removed outright, not hidden behind a default.

What this module does instead: it reports numbers with their units, the method
that produced them, and their PROVENANCE (measured on the 2D image vs. derived
from the synthetic 3D volume); it aggregates repeated reconstructions into
median + IQR; it ranks samples against each other, which needs no thresholds;
and it will apply pass/fail bands ONLY if the user supplies them together with
a citation for where they came from.

Public API
----------
    volume_fractions_3d(vol, names)            -> {phase: fraction}
    build_metric_rows(...)                     -> [MetricRow]
    metrics_record(...)                        -> flat dict, one saved run
    aggregate_runs(runs)                       -> [{Sample, Metric, n, median, ...}]
    rank_samples(aggregated)                   -> [{Metric, Sample, median, rank}]
    apply_user_criteria(rows, criteria)        -> [MetricRow-with-judgement dicts]
    STANDING_CAVEATS, DEFINITIONS
"""

from __future__ import annotations

import datetime as _dt
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ----------------------------------------------------------------------------
# Schema / provenance vocabulary
# ----------------------------------------------------------------------------
SCHEMA_VERSION = 2

PROV_2D = "2D-measured (SEM image)"
PROV_3D = "synthetic-3D (DREAM.3D, from 2D statistics)"
PROV_INPUT = "input / setting"

DIR_HIGHER = "higher"   # larger value = more favourable for transport
DIR_LOWER = "lower"     # smaller value = more favourable for transport
DIR_NONE = None         # no defensible direction; ranked only on request


# ----------------------------------------------------------------------------
# Standing caveats — shown on screen and written into every export
# ----------------------------------------------------------------------------
STANDING_CAVEATS: List[str] = [
    "Rows tagged 'synthetic-3D' are measured on a DREAM.3D volume generated "
    "from 2D statistics, not on the real 3D microstructure. They inherit "
    "whatever error the 2D→3D synthesis introduces and should be reported as "
    "synthetic-derived quantities.",
    "Rows tagged '2D-measured' come directly from the segmented micrograph and "
    "are independent of the synthesis step. Where a 2D and a synthetic-3D "
    "version of the same quantity are both listed, their difference is a "
    "free internal consistency check.",
    "Percolation and tortuosity are reported as measured along the chosen "
    "transport axis. A non-percolating phase is reported as such; it is not "
    "scored, because whether that is a defect depends on the sample's state.",
    "Two different TPB definitions exist in this pipeline: the voxel-EDGE "
    "length density reported here, and the 3×3×3 neighbourhood voxel COUNT "
    "shown under the Stage 5 boundary figures. They are not interchangeable. "
    "State which one you report.",
    "Segmentation thresholds, watershed on/off, and connectivity choice all "
    "change these numbers. The settings used are recorded with each run.",
    "This tool applies no quality thresholds of its own. Any pass/fail bands "
    "shown were entered by the user and are attributed to the user's source.",
]


# ----------------------------------------------------------------------------
# Plain-language definitions (also used by the in-app glossary)
# ----------------------------------------------------------------------------
DEFINITIONS: Dict[str, str] = {
    "volume_fraction": (
        "**Volume fraction** — the share of the structure occupied by one phase. "
        "Reported twice where possible: the 2D areal fraction measured on the "
        "segmented micrograph, and the 3D fraction of the synthesized volume. "
        "Under a stereological assumption the two should agree; a large gap is "
        "evidence the synthesis did not reproduce the input statistics."),
    "percolation": (
        "**Percolation** — whether a phase forms at least one connected path "
        "from the entry face to the exit face along the chosen transport axis. "
        "Reported as measured. A phase that does not percolate cannot conduct "
        "across the volume, but whether that is expected depends on the sample "
        "(for example, an unreduced electrode need not have a percolating pore "
        "network)."),
    "tortuosity": (
        "**Geometric tortuosity (τ)** — shortest connected path through a phase "
        "divided by the straight-line distance, by breadth-first search on the "
        "voxel grid with face connectivity. τ = 1 is a straight channel. This is "
        "a purely geometric quantity and is a lower bound on the diffusive "
        "tortuosity that appears in transport models. Undefined for a "
        "non-percolating phase."),
    "connectivity": (
        "**Connectivity ratio** — the fraction of a phase's voxels belonging to "
        "its single largest connected cluster. Voxels outside that cluster are "
        "isolated islands that contribute volume but no transport."),
    "tpb_density": (
        "**Triple-phase boundary (TPB) density** — total length of the lines "
        "where all three phases meet, divided by the analyzed volume "
        "(µm/µm³ = µm⁻²), by the voxel-edge method. These lines host the "
        "electrochemistry in a composite electrode. Note this counts *geometric* "
        "TPB; the electrochemically active subset (percolating on all three "
        "phases) is smaller."),
    "dpb_area": (
        "**Double-phase boundary (DPB) interfacial area** — contact area between "
        "each pair of phases (µm²), from counting shared voxel faces × face area. "
        "Voxel-face area counting overestimates the area of a smooth surface by a "
        "roughly constant factor, so compare DPB areas between samples rather "
        "than treating them as absolute surface areas."),
    "replicates": (
        "**Replicates** — repeated DREAM.3D reconstructions from the same 2D "
        "statistics. The synthesis is stochastic, so a single run is one draw "
        "from a distribution. Save several runs per sample and report the median "
        "with its interquartile range; the spread is the reconstruction noise "
        "floor that any between-sample difference has to clear."),
}


# ----------------------------------------------------------------------------
# Metric rows
# ----------------------------------------------------------------------------
@dataclass
class MetricRow:
    """One reported quantity: what it is, what it measured, where it came from."""
    key: str                       # stable machine key, e.g. 'tau__Pore'
    label: str                     # display label
    value: Any                     # float, bool, or None
    unit: str
    method: str                    # how it was computed
    provenance: str                # PROV_2D / PROV_3D / PROV_INPUT
    direction: Optional[str] = DIR_NONE
    note: str = ""

    def display_value(self) -> str:
        v = self.value
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            if math.isnan(v):
                return "—"
            if v != 0 and abs(v) < 1e-3:
                return f"{v:.3e}"
            return f"{v:.4g}"
        return str(v)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Metric": self.label,
            "Value": self.display_value(),
            "Unit": self.unit,
            "Method": self.method,
            "Provenance": self.provenance,
            "Note": self.note,
        }


def build_metric_rows(
    tortuosity_rows: Sequence[dict],
    boundary_result,
    vf3d: Dict[str, float],
    vf2d: Optional[Dict[str, float]] = None,
    psds: Optional[Sequence[Any]] = None,
    tpb_voxel_count: Optional[int] = None,
) -> List[MetricRow]:
    """Assemble every reported quantity for the current sample.

    Parameters
    ----------
    tortuosity_rows : output of ``psm.tortuosity.all_phase_tortuosity``.
    boundary_result : ``psm.boundaries.BoundaryResult``.
    vf3d : phase -> volume fraction measured on the synthetic 3D volume.
    vf2d : phase -> areal fraction from the 2D segmentation (Stage 2), optional.
           Included so the 2D input and the 3D output of the synthesis sit in
           the same table and can be compared by eye.
    psds : list of ``psm.psd`` results, optional; supplies the lognormal fit
           parameters that were actually fed to DREAM.3D.
    tpb_voxel_count : the 3×3×3 neighbourhood TPB voxel count from Stage 5,
           optional; reported alongside the voxel-edge density as a
           cross-check, never as the same number.
    """
    rows: List[MetricRow] = []

    # --- volume fractions: 2D input, then 3D synthetic output ---------------
    if vf2d:
        for phase, f in vf2d.items():
            rows.append(MetricRow(
                key=f"vf2d__{phase}",
                label=f"Volume fraction (2D areal) — {phase}",
                value=float(f), unit="—",
                method="classified-pixel count / total classified pixels",
                provenance=PROV_2D, direction=DIR_NONE))

    for phase, f in vf3d.items():
        note = ""
        if vf2d and phase in vf2d:
            d = float(f) - float(vf2d[phase])
            note = f"synthetic minus 2D = {d:+.3f}"
        rows.append(MetricRow(
            key=f"vf3d__{phase}",
            label=f"Volume fraction (3D) — {phase}",
            value=float(f), unit="—",
            method="voxel count / total voxels",
            provenance=PROV_3D, direction=DIR_NONE, note=note))

    # --- percolation / tortuosity / connectivity ----------------------------
    for r in tortuosity_rows:
        phase = str(r.get("Phase", "?"))
        percolates = bool(r.get("Percolates"))
        rows.append(MetricRow(
            key=f"perc__{phase}", label=f"Percolates — {phase}",
            value=percolates, unit="yes/no",
            method="connected component spans the transport axis",
            provenance=PROV_3D, direction=DIR_NONE))

    for r in tortuosity_rows:
        phase = str(r.get("Phase", "?"))
        percolates = bool(r.get("Percolates"))
        tau = float(r.get("Tortuosity", float("nan")))
        rows.append(MetricRow(
            key=f"tau__{phase}", label=f"Geometric tortuosity τ — {phase}",
            value=tau if (percolates and math.isfinite(tau)) else None,
            unit="—", method="BFS shortest path / straight-line distance",
            provenance=PROV_3D, direction=DIR_LOWER,
            note="" if percolates else "undefined — phase does not percolate"))

    for r in tortuosity_rows:
        phase = str(r.get("Phase", "?"))
        rows.append(MetricRow(
            key=f"conn__{phase}", label=f"Connectivity ratio — {phase}",
            value=float(r.get("ConnectivityRatio", float("nan"))), unit="—",
            method="largest-cluster voxels / phase voxels",
            provenance=PROV_3D, direction=DIR_HIGHER))

    # --- boundaries ---------------------------------------------------------
    rows.append(MetricRow(
        key="tpb_density", label="TPB density (voxel-edge)",
        value=float(boundary_result.tpb_density_um_per_um3), unit="µm⁻²",
        method="triple-line edge length / analyzed volume",
        provenance=PROV_3D, direction=DIR_HIGHER,
        note="geometric TPB; the electrochemically active subset is smaller"))
    rows.append(MetricRow(
        key="tpb_length", label="TPB total length (voxel-edge)",
        value=float(boundary_result.tpb_length_um), unit="µm",
        method="sum of triple-line voxel edges",
        provenance=PROV_3D, direction=DIR_NONE))

    if tpb_voxel_count is not None:
        rows.append(MetricRow(
            key="tpb_voxels_3x3x3", label="TPB voxel count (3×3×3)",
            value=int(tpb_voxel_count), unit="voxels",
            method="3×3×3 neighbourhood containing all three phases",
            provenance=PROV_3D, direction=DIR_NONE,
            note="different definition from the voxel-edge density above; "
                 "cross-check only, do not report both as one quantity"))

    rows.append(MetricRow(
        key="volume_um3", label="Analyzed volume",
        value=float(boundary_result.total_volume_um3), unit="µm³",
        method="voxel count × voxel volume",
        provenance=PROV_3D, direction=DIR_NONE))

    for (a, b), area in boundary_result.dpb_area_um2.items():
        rows.append(MetricRow(
            key=f"dpb__{a}-{b}", label=f"DPB interfacial area — {a}–{b}",
            value=float(area), unit="µm²",
            method="shared voxel faces × face area",
            provenance=PROV_3D, direction=DIR_NONE))

    # --- the 2D statistics that were actually fed to the synthesis ----------
    if psds:
        for p in psds:
            name = getattr(p, "name", "?")
            mu = getattr(p, "lognormal_mu", float("nan"))
            sig = getattr(p, "lognormal_sigma", float("nan"))
            rows.append(MetricRow(
                key=f"psd_mu__{name}", label=f"Lognormal µ (ln d) — {name}",
                value=float(mu), unit="ln(µm)",
                method="lognormal fit to the 2D particle-size distribution",
                provenance=PROV_2D, direction=DIR_NONE))
            rows.append(MetricRow(
                key=f"psd_sigma__{name}", label=f"Lognormal σ (ln d) — {name}",
                value=float(sig), unit="ln(µm)",
                method="lognormal fit to the 2D particle-size distribution",
                provenance=PROV_2D, direction=DIR_NONE))

    return rows


# ----------------------------------------------------------------------------
# One saved run
# ----------------------------------------------------------------------------
def metrics_record(sample_id: str,
                   rows: Sequence[MetricRow],
                   replicate: int = 1,
                   provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten one analysis into a single saved run.

    No thresholds and no calibration are applied: values are exactly what the
    pipeline computed. Provenance keys are prefixed ``run_`` so a CSV of many
    runs stays flat and self-describing.
    """
    rec: Dict[str, Any] = {
        "Sample": str(sample_id),
        "Replicate": int(replicate),
        "RunID": uuid.uuid4().hex[:8],
        "Timestamp": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "SchemaVersion": SCHEMA_VERSION,
    }
    for r in rows:
        v = r.value
        if isinstance(v, float) and math.isfinite(v):
            v = round(v, 6)
        rec[r.key] = v
    for k, v in (provenance or {}).items():
        rec[f"run_{k}"] = v
    return rec


def metric_directions(rows: Sequence[MetricRow]) -> Dict[str, Optional[str]]:
    """Map metric key -> favourable direction, for the ranking table."""
    return {r.key: r.direction for r in rows}


def metric_labels(rows: Sequence[MetricRow]) -> Dict[str, str]:
    return {r.key: r.label for r in rows}


# ----------------------------------------------------------------------------
# Replicate aggregation
# ----------------------------------------------------------------------------
def _numeric(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def aggregate_runs(runs: Sequence[dict],
                   labels: Optional[Dict[str, str]] = None) -> List[dict]:
    """Median / IQR / n per (sample, metric) across saved replicates.

    Median and IQR rather than mean and SD: n is small, the reconstruction
    distribution is not assumed normal, and a single bad run should not drag
    the reported centre.
    """
    import statistics as _stats

    labels = labels or {}
    meta = {"Sample", "Replicate", "RunID", "Timestamp", "SchemaVersion"}
    keys: List[str] = []
    for r in runs:
        for k in r:
            if k not in meta and not k.startswith("run_") and k not in keys:
                keys.append(k)

    by_sample: Dict[str, List[dict]] = {}
    for r in runs:
        by_sample.setdefault(str(r.get("Sample", "?")), []).append(r)

    out: List[dict] = []
    for sample, rs in by_sample.items():
        for k in keys:
            vals = [x for x in (_numeric(r.get(k)) for r in rs) if x is not None]
            n_missing = len(rs) - len(vals)
            if not vals:
                out.append({"Sample": sample, "MetricKey": k,
                            "Metric": labels.get(k, k), "n": 0,
                            "median": float("nan"), "q25": float("nan"),
                            "q75": float("nan"), "IQR": float("nan"),
                            "min": float("nan"), "max": float("nan"),
                            "n_undefined": n_missing})
                continue
            vs = sorted(vals)
            if len(vs) >= 4:
                qs = _stats.quantiles(vs, n=4, method="inclusive")
                q25, q75 = qs[0], qs[2]
            else:
                q25, q75 = vs[0], vs[-1]
            out.append({
                "Sample": sample, "MetricKey": k, "Metric": labels.get(k, k),
                "n": len(vs), "median": _stats.median(vs),
                "q25": q25, "q75": q75, "IQR": q75 - q25,
                "min": vs[0], "max": vs[-1], "n_undefined": n_missing,
            })
    return out


def rank_samples(aggregated: Sequence[dict],
                 directions: Dict[str, Optional[str]]) -> List[dict]:
    """Rank samples against each other, metric by metric.

    Ranking needs no thresholds, which is why it is the only comparative
    judgement this module makes. Rank 1 is the most transport-favourable value
    given the metric's direction. Metrics with no defensible direction are
    ranked descending by value and flagged, so the ordering is not read as a
    quality ordering.
    """
    by_metric: Dict[str, List[dict]] = {}
    for row in aggregated:
        if row["n"] > 0:
            by_metric.setdefault(row["MetricKey"], []).append(row)

    out: List[dict] = []
    for key, rows in by_metric.items():
        d = directions.get(key, DIR_NONE)
        reverse = (d != DIR_LOWER)          # higher-is-better and no-direction
        ordered = sorted(rows, key=lambda r: r["median"], reverse=reverse)
        for i, r in enumerate(ordered, start=1):
            out.append({
                "Metric": r["Metric"], "MetricKey": key, "Sample": r["Sample"],
                "median": r["median"], "IQR": r["IQR"], "n": r["n"],
                "Rank": i,
                "Ranked by": {DIR_LOWER: "lower = rank 1",
                              DIR_HIGHER: "higher = rank 1"}.get(
                                  d, "descending (no favourable direction)"),
            })
    return out


def spread_overlap_note(aggregated: Sequence[dict], metric_key: str) -> str:
    """Say plainly whether two samples' replicate ranges overlap on one metric.

    This is a descriptive statement about intervals, not a hypothesis test —
    the formal test belongs in the study's statistics, not in the tool.
    """
    rows = [r for r in aggregated if r["MetricKey"] == metric_key and r["n"] > 1]
    if len(rows) < 2:
        return ""
    rows = sorted(rows, key=lambda r: r["median"])
    overlapping = []
    for a, b in zip(rows, rows[1:]):
        if a["max"] >= b["min"]:
            overlapping.append(f"{a['Sample']}/{b['Sample']}")
    if not overlapping:
        return ("Replicate ranges do not overlap between any adjacent pair on "
                "this metric.")
    return ("Replicate ranges overlap for: " + ", ".join(overlapping) +
            ". Overlapping ranges mean the separation is not larger than the "
            "reconstruction noise for these runs.")


# ----------------------------------------------------------------------------
# Opt-in user criteria
# ----------------------------------------------------------------------------
@dataclass
class UserCriterion:
    """A pass/fail band the USER supplied, with the source they got it from."""
    metric_key: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    source: str = ""

    def is_valid(self) -> tuple[bool, str]:
        if self.lo is None and self.hi is None:
            return False, "Give at least a lower or an upper bound."
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            return False, "Lower bound is above the upper bound."
        if not self.source.strip():
            return False, ("Enter the source for this band (paper, standard, or "
                           "'own judgement'). Criteria are recorded with their "
                           "attribution.")
        return True, ""

    def describe(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"{self.lo:g} – {self.hi:g}"
        if self.lo is not None:
            return f"≥ {self.lo:g}"
        return f"≤ {self.hi:g}"

    def judge(self, value: Any) -> str:
        v = _numeric(value)
        if v is None:
            return "—"
        if self.lo is not None and v < self.lo:
            return "outside"
        if self.hi is not None and v > self.hi:
            return "outside"
        return "within"


def apply_user_criteria(rows: Sequence[MetricRow],
                        criteria: Sequence[UserCriterion]) -> List[dict]:
    """Attach the user's own bands to the metric table. Empty input -> empty."""
    by_key = {c.metric_key: c for c in criteria if c.is_valid()[0]}
    out: List[dict] = []
    for r in rows:
        c = by_key.get(r.key)
        if c is None:
            continue
        out.append({
            "Metric": r.label,
            "Value": r.display_value(),
            "Unit": r.unit,
            "User criterion": c.describe(),
            "Result": c.judge(r.value),
            "Criterion source (user-supplied)": c.source.strip(),
        })
    return out


# ----------------------------------------------------------------------------
# Kept from the previous module (still used by Stage 5)
# ----------------------------------------------------------------------------
def volume_fractions_3d(vol, names: Dict[int, str]) -> Dict[str, float]:
    """Phase volume fractions measured directly on the 3D volume."""
    import numpy as np
    total = vol.size
    return {names[lab]: float(np.count_nonzero(vol == lab)) / total
            for lab in sorted(names)}