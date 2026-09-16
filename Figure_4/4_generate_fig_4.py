"""
run_figure4.py

Generates Panel (b) data only: Phi_S(B0) at (theta, phi) = (pi/4, pi/4),
for the four polarisation preparations (P0, Px, Py, Pz).

Panel (a) reuses figure2_data.npz's Bz_P0/Bz_Px/Bz_Py/Bz_Pz curves
directly (B parallel z, already computed), so it is not regenerated here.

Same fixed physical parameters as Figures 1-3. Parallelised across
processes: one worker per preparation.
"""
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import os
# Prevent BLAS/OpenMP oversubscription: each worker process should use a
# single thread, since parallelism is across processes, not within them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (must match Figures 1-3)
# ----------------------------------------------------------------------
D_SPINS = [0.5]
A_SPINS = []

A_TENSOR_D = [np.diag([1.0, -0.25, -0.5])]
A_TENSOR_A = []

D_TENSOR = None
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match Figures 1-3
K_T = 1.0                 # PLACEHOLDER: must match Figures 1-3
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

# Off-axis field orientation
THETA, PHI = np.pi / 4, np.pi / 4

# ----------------------------------------------------------------------
# B0 sweep (PLACEHOLDER: must match Figures 1-3 for a consistent panel)
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000

OUTPUT_FILE = "Figure_4/figure4_offaxis_data.npz"

N_WORKERS = 4   # one preparation per worker

# Polarisation preparations: (p_val, pol_axis)
PREPARATIONS = {
    "P0": (0.0, 'z'),   # axis irrelevant when p_val = 0
    "Px": (1.0, 'x'),
    "Py": (1.0, 'y'),
    "Pz": (1.0, 'z'),
}

# ----------------------------------------------------------------------
_worker_builder = None


def _init_worker():
    global _worker_builder
    _worker_builder = RPMBuilder(
        d_spins=D_SPINS, a_spins=A_SPINS,
        a_tensor_d=A_TENSOR_D, a_tensor_a=A_TENSOR_A,
        d_tensor=D_TENSOR, j_ex=J_EX,
        k_s=K_S, k_t=K_T, k_r=K_R,
        use_ciss=USE_CISS, chi_percent=CHI_PERCENT
    )


def _compute_curve(task):
    """Runs the full B0 sweep for one preparation on a single worker."""
    prep_label, p_val, pol_axis, B0_values = task
    curve = np.zeros(len(B0_values))
    for i, B0 in enumerate(B0_values):
        S_sh, Tp_sh, T0_sh, Tm_sh = _worker_builder.yield_(
            B0, p_val=p_val, pol_axis=pol_axis, theta=THETA, phi=PHI
        )
        curve[i] = S_sh
    return prep_label, curve


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    tasks = [
        (prep_label, p_val, pol_axis, B0_values)
        for prep_label, (p_val, pol_axis) in PREPARATIONS.items()
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as executor:
        for prep_label, curve in tqdm(
            executor.map(_compute_curve, tasks), total=len(tasks)
        ):
            results[prep_label] = curve

    np.savez(OUTPUT_FILE, B0_values=B0_values, **results)
    print(f"Saved to {OUTPUT_FILE}")