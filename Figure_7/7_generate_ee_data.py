"""
run_figure7_offaxis_jd.py

Replaces run_jd_robustness.py's B-parallel-z, Pz-only test. Same four
panels, same J_VALUE/D0_VALUE placeholders, but at the off-axis field
where the transverse response exists, reporting c_x, c_y, c_z rather
than just Pz = +1, 0, -1:

    (a) H_Z + H_hf                  (baseline, J=0, D=0)
    (b) H_Z + H_hf + H_ex            (J != 0, D=0)
    (c) H_Z + H_hf + H_dip           (J=0, D != 0)
    (d) H_Z + H_hf + H_ex + H_dip    (J != 0, D != 0)

J and D0 are PLACEHOLDERS (0.1 mT each), copied from the original
z-aligned test for continuity -- replace with whatever values are
already established/justified elsewhere in the thesis if those exist.

One task per (panel, preparation) curve: 16 tasks total.
"""
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (must match Figures 1-4's one-nucleus setup)
# ----------------------------------------------------------------------
A1 = np.diag([1.0, -0.25, -0.5])

K_S = 1.0                 # PLACEHOLDER: must match Figures 1-4
K_T = 1.0                 # PLACEHOLDER: must match Figures 1-4
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA_FIELD, PHI_FIELD = np.pi / 4, np.pi / 4   # off-axis, matches Figure 4

# PLACEHOLDER: replace with established values if available, rather
# than treating 0.1 mT as a considered choice for either parameter.
J_VALUE = 0.1    # mT
D0_VALUE = 0.1   # mT
D_TENSOR_ON = D0_VALUE * np.diag([-2/3, -2/3, 4/3])

# ----------------------------------------------------------------------
# B0 sweep (PLACEHOLDER: must match Figures 1-4)
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000

OUTPUT_FILE = "Figure_7/figure7_offaxis_jd_data.npz"

N_WORKERS = 16   # one curve per worker: 4 panels x 4 preparations

# ----------------------------------------------------------------------
PANELS = {
    "a_baseline":  {"j_ex": 0.0,      "d_tensor": None},
    "b_exchange":  {"j_ex": J_VALUE,  "d_tensor": None},
    "c_dipolar":   {"j_ex": 0.0,      "d_tensor": D_TENSOR_ON},
    "d_both":      {"j_ex": J_VALUE,  "d_tensor": D_TENSOR_ON},
}

PREPARATIONS = {
    "P0": [0.0, 0.0, 0.0],
    "Px": [1.0, 0.0, 0.0],
    "Py": [0.0, 1.0, 0.0],
    "Pz": [0.0, 0.0, 1.0],
}

# ----------------------------------------------------------------------
_worker_builder = None


def _init_worker(panel_key):
    global _worker_builder
    panel = PANELS[panel_key]
    _worker_builder = RPMBuilder(
        d_spins=[0.5], a_spins=[],
        a_tensor_d=[A1], a_tensor_a=[],
        d_tensor=panel["d_tensor"], j_ex=panel["j_ex"],
        k_s=K_S, k_t=K_T, k_r=K_R,
        use_ciss=USE_CISS, chi_percent=CHI_PERCENT
    )


def _compute_curve(task):
    panel_key, prep_label, p_vec, B0_values = task
    curve = np.zeros(len(B0_values))
    for i, B0 in enumerate(B0_values):
        rho0 = _worker_builder.initial_state_custom(p_d=[p_vec], p_a=[])
        S_sh, Tp_sh, T0_sh, Tm_sh = _worker_builder.yield_(
            B0, theta=THETA_FIELD, phi=PHI_FIELD, rho0=rho0
        )
        curve[i] = S_sh
    return panel_key, prep_label, curve


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    results = {}

    for panel_key in PANELS:
        tasks = [
            (panel_key, prep_label, p_vec, B0_values)
            for prep_label, p_vec in PREPARATIONS.items()
        ]
        with ProcessPoolExecutor(
            max_workers=min(N_WORKERS, len(tasks)),
            initializer=_init_worker,
            initargs=(panel_key,),
        ) as executor:
            for _, prep_label, curve in tqdm(
                executor.map(_compute_curve, tasks),
                total=len(tasks),
                desc=panel_key,
            ):
                results[f"{panel_key}_{prep_label}"] = curve

    np.savez(OUTPUT_FILE, B0_values=B0_values, **results)
    print(f"Saved to {OUTPUT_FILE}")

    # Diagnostic: max|c_alpha| per panel, to quantify whether J/D
    # suppress, enhance, or leave untouched each transverse component
    # individually -- something the old Pz-only test couldn't show.
    print()
    for panel_key in PANELS:
        p0 = results[f"{panel_key}_P0"]
        cx = results[f"{panel_key}_Px"] - p0
        cy = results[f"{panel_key}_Py"] - p0
        cz = results[f"{panel_key}_Pz"] - p0
        print(f"{panel_key}: max|c_x|={np.abs(cx).max():.4e}, "
              f"max|c_y|={np.abs(cy).max():.4e}, max|c_z|={np.abs(cz).max():.4e}")