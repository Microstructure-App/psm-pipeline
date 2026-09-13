"""H3 step 2: synthesize one 3D volume per (slice x replicate) via the app's
own DREAM.3D path. Run on the Windows machine with PipelineRunner 6.5.171.

Place this file in the repo root next to psm/ and h3_2d_stats.json.

Usage:  python h3_generate.py --replicates 5
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)
import argparse
import json
import os
from pathlib import Path

from psm import dream3d_runner as dr

TEMPLATE = "3D_generator_simple.json"
RUNNER = r"C:\DREAM3D\DREAM3D-6.5.171-Win64\PipelineRunner.exe"
OUT = Path("h3_outputs")
NAMES = ["Pore", "LSCF", "GDC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--stats", default="h3_2d_stats.json")
    a = ap.parse_args()

    spec = json.load(open(a.stats))
    nx, ny, nz = spec["target_dims_xyz"]          # 120, 283, 151
    res = spec["voxel_um"]                        # 0.040
    OUT.mkdir(exist_ok=True)
    manifest = []
    for rec in spec["records"]:
        z = rec["slice"]
        phases = [dr.PhaseParams(phase_fraction=float(rec[f"VF_{n}"]),
                                 mu=float(rec[f"mu_{n}"]),
                                 sigma=float(rec[f"sigma_{n}"]),
                                 name=n) for n in NAMES]
        for r in range(a.replicates):
            vtk = str(OUT / f"h3_slice{z}_rep{r}.vtk")
            if os.path.exists(vtk):
                # resume-safe WITHOUT stale mixing: inputs are fixed by
                # h3_2d_stats.json, so an existing file is by construction
                # computed from the same parameters. If you ever change the
                # stats file, DELETE h3_outputs/ first.
                print("skip (exists):", vtk)
                manifest.append(dict(slice=z, rep=r, vtk=vtk))
                continue
            print(f"slice {z} rep {r} ...")
            dr.generate_microstructure(template_path=TEMPLATE, phases=phases,
                                       output_vtk=vtk,
                                       pipeline_runner_exe=RUNNER,
                                       dims=(nx, ny, nz), resolution_um=res,
                                       stream=True)
            manifest.append(dict(slice=z, rep=r, vtk=vtk))
    json.dump(manifest, open(OUT / "h3_manifest.json", "w"), indent=1)
    print(f"Done: {len(manifest)} volumes.")


if __name__ == "__main__":
    main()
