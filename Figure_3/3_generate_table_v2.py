"""
run_figure3_table_v2.py

At each of the three marked resonance fields, finds the pair of
near-degenerate active-manifold eigenstates of H, then computes the
full ordered-pair diagnostic set for Pz = +1, 0, -1:

    w_i    = <Ei|rho0|Ei>
    w_j    = <Ej|rho0|Ej>
    rho_ij = <Ei|rho0|Ej>
    S_ji   = <Ej|P_S|Ei>
    2Re[rho_ij * S_ji]
    C_ij   = 2 Re[ k * rho_ij * S_ji / (k + i*(Ei-Ej)) ]

Replaces run_figure3_table.py's W = w_i + w_j, which could look large
even when only one of the two states was populated. C_ij is the exact
equal-rate interference contribution to the singlet yield; it is only
computed when K_S == K_T (asserted below), and left blank otherwise.

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

OUTPUT_FILE = "Figure_3/figure3_table_v2.csv"

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
P_S = builder.pop_ops['S']

EQUAL_RATES = np.isclose(K_S, K_T)
if not EQUAL_RATES:
    print("NOTE: K_S != K_T -- C_ij is only valid for equal rates and "
          "will be left blank for this run.")


def find_active_pair(B0):
    """
    Diagonalises H at B0, restricts to active-manifold eigenstates
    (active population weight > threshold), and returns the two
    eigenstates with the smallest energy gap among that subset,
    along with their eigenvalues.
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

    return (ekets[i], ekets[j], sorted_evals[min_gap_pos],
            sorted_evals[min_gap_pos + 1], i, j)


def scalar(x):
    """Extract a python complex from a 1x1 Qobj or bare number."""
    return x if np.isscalar(x) else complex(x.full()[0, 0])


def pair_diagnostics(ket_i, ket_j, Ei, Ej, p_val):
    rho0 = builder.initial_state(p_val=p_val, pol_axis='z')

    assert rho0.dims == P_S.dims, (
        f"Dimension mismatch: rho0.dims={rho0.dims}, P_S.dims={P_S.dims}. "
        "P_S must be embedded in the full electron-nuclear space."
    )

    w_i = np.real(scalar(ket_i.dag() * rho0 * ket_i))
    w_j = np.real(scalar(ket_j.dag() * rho0 * ket_j))
    rho_ij = scalar(ket_i.dag() * rho0 * ket_j)
    S_ji = scalar(ket_j.dag() * P_S * ket_i)
    interference = 2 * np.real(rho_ij * S_ji)

    C_ij = ""
    if EQUAL_RATES:
        delta_E = Ei - Ej
        C_ij = 2 * np.real(K_S * rho_ij * S_ji / (K_S + 1j * delta_E))

    return w_i, w_j, abs(rho_ij), abs(S_ji), interference, C_ij


rows = []
for B0 in RESONANCE_B0:
    ket_i, ket_j, Ei, Ej, i, j = find_active_pair(B0)
    gap = Ei - Ej
    print(f"B0={B0} mT: nearest active pair indices ({i}, {j}), gap={gap:.6e}")

    for p_val, label in [(+1.0, "plus1"), (0.0, "0"), (-1.0, "minus1")]:
        w_i, w_j, abs_rho_ij, abs_S_ji, interference, C_ij = pair_diagnostics(
            ket_i, ket_j, Ei, Ej, p_val
        )
        rows.append({
            "B0_mT": B0,
            "eigenstate_i": i,
            "eigenstate_j": j,
            "energy_gap": gap,
            "Pz": label,
            "w_i": w_i,
            "w_j": w_j,
            "abs_rho_ij": abs_rho_ij,
            "abs_S_ji": abs_S_ji,
            "2Re_rho_ij_S_ji": interference,
            "C_ij": C_ij,
        })

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {OUTPUT_FILE}")
for row in rows:
    print(row)