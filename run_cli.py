"""
Command-line runner for the PCFC microstructure pipeline.

Examples
--------
# Segment by explicit thresholds, run PSD, and write a DREAM.3D pipeline:
python run_cli.py analyze --image OFA8_SEM.png --scale-um 30 --scale-px 200 \
    --thresholds "Pore:0-40,NiO:60-130,BZCYYb:150-255" \
    --template 3phasemicrostructure_stl_removeisolated.json --out results

# Or let multi-Otsu suggest thresholds for N named phases:
python run_cli.py analyze --image OFA8_SEM.png --scale-um 30 --scale-px 200 \
    --auto "Pore,NiO,BZCYYb" --out results

# Analyze a DREAM.3D output volume (tortuosity + TPB/DPB):
python run_cli.py volume --input results/microstructure.dream3d --voxel 0.1 --axis 0 \
    --names "Pore,NiO,BZCYYb"
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from psm import segmentation as seg
from psm import psd as psdmod
from psm import dream3d as d3d
from psm.tortuosity import all_phase_tortuosity
from psm.boundaries import analyze_boundaries


def _parse_thresholds(spec: str) -> list:
    """'Pore:0-40,NiO:60-130' -> [PhaseThreshold(...), ...]."""
    out = []
    for part in spec.split(","):
        name, rng = part.split(":")
        lo, hi = rng.split("-")
        out.append(seg.PhaseThreshold(name.strip(), int(lo), int(hi)))
    return out


def cmd_analyze(a: argparse.Namespace) -> None:
    os.makedirs(a.out, exist_ok=True)
    um_per_px = seg.microns_per_pixel(a.scale_px, a.scale_um)
    g8 = seg.read_sem_as_gray8(a.image)

    if a.thresholds:
        thresholds = _parse_thresholds(a.thresholds)
    elif a.auto:
        names = [n.strip() for n in a.auto.split(",")]
        thresholds = seg.suggest_thresholds_multiotsu(g8, names)
        print("Auto thresholds:", [(t.name, t.low, t.high) for t in thresholds])
    else:
        raise SystemExit("Provide --thresholds or --auto")

    res = seg.segment_by_thresholds(g8, thresholds)
    names = [t.name for t in thresholds]
    vf = seg.volume_fractions(res)
    print("Volume fractions:", {k: round(v, 4) for k, v in vf.items()})

    cfg = psdmod.PSDConfig(use_watershed=not a.no_watershed, h_max=a.h_max,
                           apply_shape_filter=a.shape_filter)
    psds = [psdmod.measure_phase(res.masks[n], um_per_px, n, cfg) for n in names]
    table = psdmod.summary_table(psds, vf)
    table.to_csv(os.path.join(a.out, "SEM_summary_table.csv"), index=False)
    print(table.to_string(index=False))

    if a.template:
        if len(psds) != 3:
            print(f"[skip pipeline] template expects 3 phases, got {len(psds)}.")
            return
        phases = [d3d.PhaseStats(name=p.name, phase_fraction=float(vf[p.name]),
                                 mu=float(p.lognormal_mu), sigma=float(p.lognormal_sigma))
                  for p in psds]
        path = d3d.write_pipeline(a.template, phases, a.out, out_prefix="microstructure",
                                  dimensions=tuple(a.dims), resolution_um=a.voxel)
        print("Wrote pipeline:", path)
        if a.dream3d_bin:
            proc = d3d.run_pipeline(path, a.dream3d_bin)
            print("PipelineRunner return code:", proc.returncode)
            if proc.returncode != 0:
                print(proc.stdout[-2000:]); print(proc.stderr[-2000:])


def cmd_volume(a: argparse.Namespace) -> None:
    vol = (d3d.load_phase_volume_h5(a.input) if a.input.endswith(".dream3d")
           else d3d.load_phase_volume_vtk(a.input))
    print("Volume (z,y,x):", vol.shape, "labels:", sorted(np.unique(vol).tolist()))
    labels = sorted(int(x) for x in np.unique(vol))
    user_names = [n.strip() for n in a.names.split(",")] if a.names else []
    names = {lab: (user_names[lab - 1] if 0 < lab <= len(user_names) else f"Phase_{lab}")
             for lab in labels}
    tort = pd.DataFrame(all_phase_tortuosity(vol, names, axis=a.axis))
    print("\nTortuosity:\n", tort.to_string(index=False))
    b = analyze_boundaries(vol, names, voxel_size_um=a.voxel)
    print("\nDPB interfacial areas (um^2):")
    for pair, area in b.dpb_area_um2.items():
        print(f"  {pair[0]}-{pair[1]}: {area:.3f}")
    print(f"TPB length = {b.tpb_length_um:.3f} um | density = "
          f"{b.tpb_density_um_per_um3:.4f} um^-2")


def main() -> None:
    p = argparse.ArgumentParser(description="PCFC microstructure pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="SEM -> segmentation + PSD + DREAM.3D pipeline")
    pa.add_argument("--image", required=True)
    pa.add_argument("--scale-um", type=float, required=True, dest="scale_um")
    pa.add_argument("--scale-px", type=float, required=True, dest="scale_px")
    pa.add_argument("--thresholds", default=None,
                    help='e.g. "Pore:0-40,NiO:60-130,BZCYYb:150-255"')
    pa.add_argument("--auto", default=None, help='comma names, e.g. "Pore,NiO,BZCYYb"')
    pa.add_argument("--no-watershed", action="store_true", dest="no_watershed")
    pa.add_argument("--h-max", type=float, default=0.3, dest="h_max")
    pa.add_argument("--shape-filter", action="store_true", dest="shape_filter")
    pa.add_argument("--template", default=None)
    pa.add_argument("--out", default="results")
    pa.add_argument("--dims", type=int, nargs=3, default=[100, 100, 50])
    pa.add_argument("--voxel", type=float, default=0.1)
    pa.add_argument("--dream3d-bin", default=None, dest="dream3d_bin")
    pa.set_defaults(func=cmd_analyze)

    pv = sub.add_parser("volume", help="Analyze a DREAM.3D output volume")
    pv.add_argument("--input", required=True)
    pv.add_argument("--voxel", type=float, default=0.1)
    pv.add_argument("--axis", type=int, default=0, choices=[0, 1, 2])
    pv.add_argument("--names", default=None, help='comma names in phase order')
    pv.set_defaults(func=cmd_volume)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
