"""
run_figure5_two_nucleus_scan.py

Extends run_figure5.py's two_on_donor case: same shared-P preparation
applied to both nuclei via builder.initial_state (not per-nucleus),
now at the off-axis field and across P0/Px/Py/Pz rather than just
Pz = +1, 0, -1 at B parallel z, and with A2 scaled relative to A1
rather than fixed equal to it.

Case lambda_L: two nuclei, both on donor, A2 = L * A1 (coaxial),
BOTH nuclei receive the same shared polarisation vector.

As in the original two_on_donor case, rho_n(P) is quadratic in the
shared P here (product of two mixed single-nucleus states with the
same P), so Phi_S(0) = [Phi_S(+1)+Phi_S(-1)]/2 is NOT expected to
hold exactly except in the lambda=0 limit, where nucleus 2 decouples
and the system reduces to the one-nucleus affine case. This is a
property of the shared-P parametrisation, not a bug -- see the
original run_figure5.py docstring.

Off-axis field, matching Figure 4: (theta, phi) = (pi/4, pi/4).

One task per (lambda, preparation) curve: 5 lambdas x 4 preparations
= 20 tasks.
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
# Fixed physical parameters (must match Figures 1-4)
# ----------------------------------------------------------------------
A1 = np.diag([1.0, -0.25, -0.5])   # mT

D_TENSOR = None
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match Figures 1-4
K_T = 1.0                 # PLACEHOLDER: must match Figures 1-4
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA_FIELD, PHI_FIELD = np.pi / 4, np.pi / 4   # off-axis, matches Figure 4

# ----------------------------------------------------------------------
# B0 sweep (PLACEHOLDER: must match Figures 1-4)
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000

OUTPUT_FILE = "Figure_5/figure5_two_nucleus_scan.npz"

LAMBDA_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]   # A2 = lambda * A1

N_WORKERS = 25   # one curve per worker: 5 lambdas x 5 preparations

# ----------------------------------------------------------------------
CASES = {
    f"lambda_{lam:.2f}": {"a_tensor_d": [A1, lam * A1]}
    for lam in LAMBDA_VALUES
}

# Shared preparation applied to both nuclei via builder.initial_state.
# Pz_minus is included alongside Pz so the midpoint/quadratic-term
# diagnostic below can be computed the same way as the original script.
PREPARATIONS = {
    "P0": (0.0, "z"),   # axis irrelevant when p_val = 0
    "Px": (1.0, "x"),
    "Py": (1.0, "y"),
    "Pz": (1.0, "z"),
    "Pz_minus": (-1.0, "z"),
}

# ----------------------------------------------------------------------
_worker_builder = None


def _init_worker(case_key):
    global _worker_builder
    case = CASES[case_key]
    _worker_builder = RPMBuilder(
        d_spins=[0.5, 0.5], a_spins=[],
        a_tensor_d=case["a_tensor_d"], a_tensor_a=[],
        d_tensor=D_TENSOR, j_ex=J_EX,
        k_s=K_S, k_t=K_T, k_r=K_R,
        use_ciss=USE_CISS, chi_percent=CHI_PERCENT
    )


def _compute_curve(task):
    """Shared p_val/pol_axis applied to BOTH nuclei via initial_state
    (same convention as the original two_on_donor case)."""
    case_key, prep_label, p_val, pol_axis, B0_values = task
    curve = np.zeros(len(B0_values))
    for i, B0 in enumerate(B0_values):
        S_sh, Tp_sh, T0_sh, Tm_sh = _worker_builder.yield_(
            B0, p_val=p_val, pol_axis=pol_axis, theta=THETA_FIELD, phi=PHI_FIELD
        )
        curve[i] = S_sh
    return case_key, prep_label, curve


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    results = {}

    for case_key in CASES:
        tasks = [
            (case_key, prep_label, p_val, pol_axis, B0_values)
            for prep_label, (p_val, pol_axis) in PREPARATIONS.items()
        ]
        with ProcessPoolExecutor(
            max_workers=min(N_WORKERS, len(tasks)),
            initializer=_init_worker,
            initargs=(case_key,),
        ) as executor:
            for _, prep_label, curve in tqdm(
                executor.map(_compute_curve, tasks),
                total=len(tasks),
                desc=case_key,
            ):
                results[f"{case_key}_{prep_label}"] = curve

    np.savez(OUTPUT_FILE, B0_values=B0_values, **results)
    print(f"Saved to {OUTPUT_FILE}")

    # Diagnostic: max|c_alpha| per lambda, and the quadratic-term
    # signature (midpoint deviation using Pz as the representative
    # axis -- expected to vanish only as lambda -> 0).
    print()
    for case_key in CASES:
        p0 = results[f"{case_key}_P0"]
        cx = results[f"{case_key}_Px"] - p0
        cy = results[f"{case_key}_Py"] - p0
        cz = results[f"{case_key}_Pz"] - p0
        print(f"{case_key}: max|c_x|={np.abs(cx).max():.4e}, "
              f"max|c_y|={np.abs(cy).max():.4e}, max|c_z|={np.abs(cz).max():.4e}")

    print()
    print("Quadratic-term check (uses Pz as representative axis; per")
    print("the docstring, this is expected to vanish only as lambda -> 0):")
    for case_key in CASES:
        p0 = results[f"{case_key}_P0"]
        pz_plus = results[f"{case_key}_Pz"]
        pz_minus = results[f"{case_key}_Pz_minus"]
        dev = np.abs(p0 - 0.5 * (pz_plus + pz_minus)).max()
        print(f"{case_key}: max midpoint deviation = {dev:.3e}")