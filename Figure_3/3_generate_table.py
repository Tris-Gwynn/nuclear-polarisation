"""
run_figure3_table.py

At each of the three marked resonance fields, finds the pair of
near-degenerate active-manifold eigenstates of H, then computes

    W(Pz) = <Ei|rho0(Pz)|Ei> + <Ej|rho0(Pz)|Ej>

for Pz = +1, 0, -1. Saves a small table (csv) rather than a full sweep.

Separate from run_figure3_main.py: this diagonalizes H directly rather
than running mesolve, and only touches the three specified fields.
"""
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import numpy as np
import qutip as qt
import csv
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (must match run_figure3_main.py)
# ----------------------------------------------------------------------
D_SPINS = [0.5]
A_SPINS = []

A_TENSOR_D = [np.diag([1.0, -0.25, -0.5])]
A_TENSOR_A = []

D_TENSOR = None
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match run_figure3_main.py
K_T = 1.0                 # PLACEHOLDER: must match run_figure3_main.py
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA, PHI = 0.0, 0.0   # B parallel z

# ----------------------------------------------------------------------
# Resonance fields to analyse (mT)
# ----------------------------------------------------------------------
RESONANCE_B0 = [0.1292, 0.3515, 0.3870]

# Minimum active-manifold population weight for an eigenstate to be
# considered "active" rather than part of the degenerate shelving block.
ACTIVE_WEIGHT_THRESHOLD = 0.5

OUTPUT_FILE = "Figure_3/figure3_table.csv"

# ----------------------------------------------------------------------
builder = RPMBuilder(
    d_spins=D_SPINS, a_spins=A_SPINS,
    a_tensor_d=A_TENSOR_D, a_tensor_a=A_TENSOR_A,
    d_tensor=D_TENSOR, j_ex=J_EX,
    k_s=K_S, k_t=K_T, k_r=K_R,
    use_ciss=USE_CISS, chi_percent=CHI_PERCENT
)

active_op = (
    builder.pop_ops['S'] + builder.pop_ops['Tp']
    + builder.pop_ops['T0'] + builder.pop_ops['Tm']
)


def find_active_pair(B0):
    """
    Diagonalises H at B0, restricts to active-manifold eigenstates
    (active population weight > threshold), and returns the two
    eigenstates with the smallest energy gap among that subset,
    along with the gap itself for diagnostic purposes.
    """
    H = builder._build_hamiltonian(B0, THETA, PHI)
    evals, ekets = H.eigenstates()

    active_mask = np.array([
        qt.expect(active_op, ket) > ACTIVE_WEIGHT_THRESHOLD for ket in ekets
    ])
    active_idx = np.where(active_mask)[0]

    if len(active_idx) < 2:
        raise RuntimeError(
            f"Fewer than 2 active eigenstates found at B0={B0} mT; "
            f"check ACTIVE_WEIGHT_THRESHOLD or the Hamiltonian construction."
        )

    active_evals = evals[active_idx]
    order = np.argsort(active_evals)
    sorted_idx = active_idx[order]
    sorted_evals = active_evals[order]

    gaps = np.diff(sorted_evals)
    min_gap_pos = np.argmin(gaps)
    i, j = sorted_idx[min_gap_pos], sorted_idx[min_gap_pos + 1]

    return ekets[i], ekets[j], gaps[min_gap_pos], i, j


def w_of_pz(B0, ket_i, ket_j, p_val):
    rho0 = builder.initial_state(p_val=p_val, pol_axis='z')
    P_ij = ket_i * ket_i.dag() + ket_j * ket_j.dag()
    return float(np.real(qt.expect(P_ij, rho0)))


rows = []
for B0 in RESONANCE_B0:
    ket_i, ket_j, gap, i, j = find_active_pair(B0)
    print(f"B0={B0} mT: nearest active pair indices ({i}, {j}), gap={gap:.6e}")

    W_plus = w_of_pz(B0, ket_i, ket_j, +1.0)
    W_zero = w_of_pz(B0, ket_i, ket_j, 0.0)
    W_minus = w_of_pz(B0, ket_i, ket_j, -1.0)

    rows.append({
        "B0_mT": B0,
        "eigenstate_i": i,
        "eigenstate_j": j,
        "energy_gap": gap,
        "W_Pz_plus1": W_plus,
        "W_Pz_0": W_zero,
        "W_Pz_minus1": W_minus,
    })

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {OUTPUT_FILE}")
for row in rows:
    print(row)