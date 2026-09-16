"""
run_rate_robustness.py

Robustness check: does the resonance-switching behaviour (Pz = +1, 0, -1)
survive when k_S != k_T, rather than the k_S = k_T = 1.0 used
throughout Figures 1-5?

Case A: baseline,      k_S=1.0, k_T=1.0
Case B: slow_triplet,  k_S=1.0, k_T=whatever RATE_KT below is set to

Everything else unchanged from Figure 3's one-nucleus setup: same
tensor, B parallel z, same B0 sweep. H is identical in both cases --
only the reaction rates entering the collapse operators change.

Parallelised at the individual (case, preparation, B0-point) level
rather than one task per curve: with only 6 curves total, curve-level
parallelism caps at 6 workers regardless of how many cores are
available. Flattening to points gives 2 x 3 x N_B0 independent tasks,
which spreads properly across many more cores.

Each worker builds BOTH case builders once at startup (cheap relative
to a single yield_ call), since a persistent worker pool serving mixed
tasks from both cases cannot commit to only one case's builder as it
could when parallelising by curve.
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

D_TENSOR = None
J_EX = 0.0
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA_FIELD, PHI_FIELD = 0.0, 0.0   # B parallel z

# ----------------------------------------------------------------------
# B0 sweep
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 500   # PLACEHOLDER: reduce further (e.g. 200) if RATE_KT is small
             # enough that per-point cost is still high even parallelised

OUTPUT_FILE = "Figure_6/rate_robustness_data.npz"

N_WORKERS = 28
CHUNKSIZE = 4   # tune down if progress reporting feels too coarse,
                 # up if IPC overhead dominates at this task count

# ----------------------------------------------------------------------
CASES = {
    "baseline":     {"k_s": 1.0, "k_t": 1.0},
    "slow_triplet": {"k_s": 1.0, "k_t": 0.1},
}

PREPARATIONS = {
    "Pz_plus":  +1.0,
    "P0":        0.0,
    "Pz_minus": -1.0,
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
    case_key, prep_label, p_val, i, B0 = task
    builder = _worker_builders[case_key]
    S_sh, Tp_sh, T0_sh, Tm_sh = builder.yield_(
        B0, p_val=p_val, pol_axis='z', theta=THETA_FIELD, phi=PHI_FIELD
    )
    return case_key, prep_label, i, S_sh


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    tasks = [
        (case_key, prep_label, p_val, i, B0)
        for case_key in CASES
        for prep_label, p_val in PREPARATIONS.items()
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

    # Diagnostic: midpoint identity should hold exactly (near machine
    # precision) in BOTH cases -- it depends only on the single nucleus
    # being spin-1/2, not on k_S vs k_T. A failure here would indicate
    # a bug, not a kinetics effect.
    for case_key in CASES:
        p0 = results[f"{case_key}_P0"]
        pz_plus = results[f"{case_key}_Pz_plus"]
        pz_minus = results[f"{case_key}_Pz_minus"]
        dev = np.abs(p0 - 0.5 * (pz_plus + pz_minus)).max()
        print(f"{case_key}: max midpoint deviation = {dev:.3e}")