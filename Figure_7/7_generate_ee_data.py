"""
run_jd_robustness.py

Final robustness figure: does the Pz-dependent resonance switching
survive when electron-electron exchange (J) and/or dipolar (D)
coupling are added to the one-nucleus baseline? This is NOT a J/D
parameter sweep -- one representative nonzero value of each, held
fixed, combined in four panels:

    (a) H_Z + H_hf                  (baseline, J=0, D=0)
    (b) H_Z + H_hf + H_ex            (J != 0, D=0)
    (c) H_Z + H_hf + H_dip           (J=0, D != 0)
    (d) H_Z + H_hf + H_ex + H_dip    (J != 0, D != 0)

Same tensor, same B0 sweep, same k_S=k_T=1.0 as the original Figures
1-4. Only the coherent Hamiltonian terms change between panels.

J and D0 are PLACEHOLDERS (0.1 mT each) -- replace with whatever
values are already established/justified elsewhere in the thesis if
those exist, rather than treating 0.1 mT as a considered choice.

One task per (panel, preparation) curve: 12 tasks total.
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

THETA_FIELD, PHI_FIELD = 0.0, 0.0   # B parallel z

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

OUTPUT_FILE = "Figure_7/jd_robustness_data.npz"

N_WORKERS = 12   # one curve per worker: 4 panels x 3 preparations

# ----------------------------------------------------------------------
PANELS = {
    "a_baseline":  {"j_ex": 0.0,      "d_tensor": None},
    "b_exchange":  {"j_ex": J_VALUE,  "d_tensor": None},
    "c_dipolar":   {"j_ex": 0.0,      "d_tensor": D_TENSOR_ON},
    "d_both":      {"j_ex": J_VALUE,  "d_tensor": D_TENSOR_ON},
}

PREPARATIONS = {
    "Pz_plus":  +1.0,
    "P0":        0.0,
    "Pz_minus": -1.0,
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
    panel_key, prep_label, p_val, B0_values = task
    curve = np.zeros(len(B0_values))
    for i, B0 in enumerate(B0_values):
        S_sh, Tp_sh, T0_sh, Tm_sh = _worker_builder.yield_(
            B0, p_val=p_val, pol_axis='z', theta=THETA_FIELD, phi=PHI_FIELD
        )
        curve[i] = S_sh
    return panel_key, prep_label, curve


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    results = {}

    for panel_key in PANELS:
        tasks = [
            (panel_key, prep_label, p_val, B0_values)
            for prep_label, p_val in PREPARATIONS.items()
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

    # Diagnostics:
    # 1. Midpoint identity should hold exactly in every panel -- it
    #    depends only on the single nucleus being spin-1/2, not on
    #    which coherent terms are present.
    # 2. Scalar summary: max_B0 |Phi_S(+1) - Phi_S(-1)| per panel, to
    #    quantify whether the polarisation dependence strengthens or
    #    weakens as J/D are added.
    print()
    for panel_key in PANELS:
        p0 = results[f"{panel_key}_P0"]
        pz_plus = results[f"{panel_key}_Pz_plus"]
        pz_minus = results[f"{panel_key}_Pz_minus"]
        midpoint_dev = np.abs(p0 - 0.5 * (pz_plus + pz_minus)).max()
        max_pol_diff = np.abs(pz_plus - pz_minus).max()
        print(f"{panel_key}: max midpoint deviation = {midpoint_dev:.3e}, "
              f"max|Phi_S(+1)-Phi_S(-1)| = {max_pol_diff:.4f}")