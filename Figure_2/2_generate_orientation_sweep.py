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
# Fixed physical parameters (must match Figure 1)
# ----------------------------------------------------------------------
D_SPINS = [0.5]          # PLACEHOLDER: donor nuclear spins
A_SPINS = []              # PLACEHOLDER: acceptor nuclear spins

A_TENSOR_D = [np.diag([1.0, -0.25, -0.5])]   # mT, one 3x3 tensor per donor nucleus
A_TENSOR_A = []                               # mT, one 3x3 tensor per acceptor nucleus

D_TENSOR = None           # J, D = 0 per spec
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match Figure 1
K_T = 1.0                 # PLACEHOLDER: must match Figure 1
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

# ----------------------------------------------------------------------
# B0 sweep (PLACEHOLDER: match the refined Figure 1 screening range)
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000

OUTPUT_FILE = "Figure_2/figure2_data.npz"

N_WORKERS = 12

# ----------------------------------------------------------------------
# Field orientations: (theta, phi) per get_n_spin_zeeman convention
#   Bx = B0 sin(theta) cos(phi)
#   By = B0 sin(theta) sin(phi)
#   Bz = B0 cos(theta)
# ----------------------------------------------------------------------
ORIENTATIONS = {
    "Bx": (np.pi / 2, 0.0),
    "By": (np.pi / 2, np.pi / 2),
    "Bz": (0.0, 0.0),
}

# Polarisation preparations: (p_val, pol_axis)
PREPARATIONS = {
    "P0": (0.0, 'z'),   # axis irrelevant when p_val = 0
    "Px": (1.0, 'x'),
    "Py": (1.0, 'y'),
    "Pz": (1.0, 'z'),
}

# ----------------------------------------------------------------------
# Worker setup: each process builds its own RPMBuilder once, rather than
# pickling a builder (with qutip Qobj c_ops/pop_ops) for every task.
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
    """Runs the full B0 sweep for one (orientation, preparation) combination
    on a single worker."""
    orient_label, prep_label, theta, phi, p_val, pol_axis, B0_values = task
    curve = np.zeros(len(B0_values))
    for i, B0 in enumerate(B0_values):
        S_sh, Tp_sh, T0_sh, Tm_sh = _worker_builder.yield_(
            B0, p_val=p_val, pol_axis=pol_axis, theta=theta, phi=phi
        )
        curve[i] = S_sh
    return orient_label, prep_label, curve


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    # One task per (orientation, preparation) combination: 12 tasks total,
    # each running its own full B0 sweep on a single worker.
    tasks = [
        (orient_label, prep_label, theta, phi, p_val, pol_axis, B0_values)
        for orient_label, (theta, phi) in ORIENTATIONS.items()
        for prep_label, (p_val, pol_axis) in PREPARATIONS.items()
    ]

    results = {}

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as executor:
        for orient_label, prep_label, curve in tqdm(
            executor.map(_compute_curve, tasks), total=len(tasks)
        ):
            results[f"{orient_label}_{prep_label}"] = curve

    np.savez(OUTPUT_FILE, B0_values=B0_values, **results)
    print(f"Saved to {OUTPUT_FILE}")