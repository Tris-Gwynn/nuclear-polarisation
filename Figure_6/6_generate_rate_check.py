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

D_TENSOR = None
J_EX = 0.0
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA_FIELD, PHI_FIELD = np.pi / 4, np.pi / 4   # off-axis, matches Figure 4

# ----------------------------------------------------------------------
# B0 sweep
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000   

OUTPUT_FILE = "Figure_6/figure6_offaxis_half_triplet_5000.npz"

N_WORKERS = 28
CHUNKSIZE = 4

# ----------------------------------------------------------------------
"""
CASES = {
    "baseline":     {"k_s": 1.0, "k_t": 1.0},
    "half_triplet": {"k_s": 1.0, "k_t": 0.5},
    "slow_triplet": {"k_s": 1.0, "k_t": 0.1},
}
"""
CASES = {
    "half_triplet": {"k_s": 1.0, "k_t": 0.5},
}

PREPARATIONS = {
    "P0": [0.0, 0.0, 0.0],
    "Px": [1.0, 0.0, 0.0],
    "Py": [0.0, 1.0, 0.0],
    "Pz": [0.0, 0.0, 1.0],
}

# ----------------------------------------------------------------------
_worker_builders = None   # dict: case_key -> RPMBuilder, built once per worker


def _init_worker():
    global _worker_builders
    _worker_builders = {
        case_key: RPMBuilder(
            d_spins=[0.5], a_spins=[],
            a_tensor_d=[A1], a_tensor_a=[],
            d_tensor=D_TENSOR, j_ex=J_EX,
            k_s=case["k_s"], k_t=case["k_t"], k_r=K_R,
            use_ciss=USE_CISS, chi_percent=CHI_PERCENT
        )
        for case_key, case in CASES.items()
    }


def _compute_point(task):
    case_key, prep_label, p_vec, i, B0 = task
    builder = _worker_builders[case_key]
    rho0 = builder.initial_state_custom(p_d=[p_vec], p_a=[])
    S_sh, Tp_sh, T0_sh, Tm_sh = builder.yield_(
        B0, theta=THETA_FIELD, phi=PHI_FIELD, rho0=rho0
    )
    return case_key, prep_label, i, S_sh


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    tasks = [
        (case_key, prep_label, p_vec, i, B0)
        for case_key in CASES
        for prep_label, p_vec in PREPARATIONS.items()
        for i, B0 in enumerate(B0_values)
    ]

    results = {
        f"{case_key}_{prep_label}": np.zeros(N_B0)
        for case_key in CASES for prep_label in PREPARATIONS
    }

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as executor:
        for case_key, prep_label, i, value in tqdm(
            executor.map(_compute_point, tasks, chunksize=CHUNKSIZE),
            total=len(tasks),
        ):
            results[f"{case_key}_{prep_label}"][i] = value

    np.savez(OUTPUT_FILE, B0_values=B0_values, **results)
    print(f"Saved to {OUTPUT_FILE}")

    for case_key in CASES:
        p0 = results[f"{case_key}_P0"]
        cx = results[f"{case_key}_Px"] - p0
        cy = results[f"{case_key}_Py"] - p0
        cz = results[f"{case_key}_Pz"] - p0
        print(f"{case_key}: max|c_x|={np.abs(cx).max():.4e}, "
              f"max|c_y|={np.abs(cy).max():.4e}, max|c_z|={np.abs(cz).max():.4e}")