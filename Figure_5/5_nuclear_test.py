"""
run_figure5_gap_tracking.py

Tests whether the eigenstate pair identified at B0=0.332 mT (via
run_figure5_donor_acceptor_sector.py) is the genuine source of the
0.332 mT feature, rather than just some pair from the correct
(exchange-symmetric) sector that would trivially show W(+1)=W(-1)=0
regardless of relevance.

The discriminating test: track the SAME two eigenstates by eigenvector
overlap continuity (not by sorted index, which can silently swap
across a sweep) over a window around 0.332 mT, and check whether their
energy gap develops a genuine local minimum there. A flat or
featureless gap across the window would mean this pair, despite
passing the sector/W checks, is not responsible for the peak.
"""
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import numpy as np
import qutip as qt
import csv
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (must match Figure 5's donor_acceptor case)
# ----------------------------------------------------------------------
A1 = np.diag([1.0, -0.25, -0.5])

D_TENSOR = None
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match Figure 5
K_T = 1.0                 # PLACEHOLDER: must match Figure 5
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA, PHI = 0.0, 0.0   # B parallel z

B0_SEED = 0.332            # field where the pair was already identified
B0_WINDOW_MIN = 0.30
B0_WINDOW_MAX = 0.36
B0_STEP = 0.001            # PLACEHOLDER: finer if the minimum needs sharper localisation

ACTIVE_WEIGHT_THRESHOLD = 0.5
SWAP_EIGENVALUE_MIN = 0.5

OUTPUT_FILE = "Figure_5/figure5_gap_tracking.csv"

# ----------------------------------------------------------------------
builder = RPMBuilder(
    d_spins=[0.5], a_spins=[0.5],
    a_tensor_d=[A1], a_tensor_a=[A1],
    d_tensor=D_TENSOR, j_ex=J_EX,
    k_s=K_S, k_t=K_T, k_r=K_R,
    use_ciss=USE_CISS, chi_percent=CHI_PERCENT
)

active_op = (
    builder.pop_ops['S'] + builder.pop_ops['Tp']
    + builder.pop_ops['T0'] + builder.pop_ops['Tm']
)


def build_swap_operator(builder):
    """Same construction as run_figure5_donor_acceptor_sector.py."""
    el_dim = builder.sys_rpm.el_dim
    P_el = np.eye(el_dim)
    P_el[:, [1, 2]] = P_el[:, [2, 1]]
    U_el = qt.Qobj(P_el, dims=[[el_dim], [el_dim]])

    U_nuc = qt.Qobj(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]]
    )
    return qt.tensor(U_el, U_nuc)


U_swap = build_swap_operator(builder)


def find_active_pair(B0):
    """Same method as run_figure5_donor_acceptor_sector.py: restrict
    to active, exchange-symmetric eigenstates, then take the minimal
    gap within that subset."""
    H = builder._build_hamiltonian(B0, THETA, PHI)
    evals, ekets = H.eigenstates()

    active_mask = np.array([
        qt.expect(active_op, ket) > ACTIVE_WEIGHT_THRESHOLD for ket in ekets
    ])
    symmetric_mask = np.array([
        qt.expect(U_swap, ket) > SWAP_EIGENVALUE_MIN for ket in ekets
    ])
    active_idx = np.where(active_mask & symmetric_mask)[0]

    if len(active_idx) < 2:
        raise RuntimeError(f"Fewer than 2 candidates at B0={B0} mT.")

    active_evals = evals[active_idx]
    order = np.argsort(active_evals)
    sorted_idx = active_idx[order]
    sorted_evals = active_evals[order]

    gaps = np.diff(sorted_evals)
    min_gap_pos = np.argmin(gaps)
    i, j = sorted_idx[min_gap_pos], sorted_idx[min_gap_pos + 1]

    return ekets[i], ekets[j], gaps[min_gap_pos]


def nearest_eigenstate(target_ket, ekets):
    """Returns the index of the eigenstate in ekets with maximum
    overlap probability with target_ket -- tracks identity by
    continuity rather than by sorted energy index, which can swap
    across an avoided crossing."""
    overlaps = [abs(target_ket.overlap(ket)) ** 2 for ket in ekets]
    return int(np.argmax(overlaps))


def track_direction(B0_list, ket_i_seed, ket_j_seed):
    """Walks outward from the seed field, tracking the two eigenstates
    by overlap continuity at each step, returning the (B0, gap) trace."""
    trace = []
    ket_i_prev, ket_j_prev = ket_i_seed, ket_j_seed
    for B0 in B0_list:
        H = builder._build_hamiltonian(B0, THETA, PHI)
        evals, ekets = H.eigenstates()

        idx_i = nearest_eigenstate(ket_i_prev, ekets)
        idx_j = nearest_eigenstate(ket_j_prev, ekets)

        E_i, E_j = evals[idx_i], evals[idx_j]
        gap = abs(E_i - E_j)
        trace.append((B0, gap))

        ket_i_prev, ket_j_prev = ekets[idx_i], ekets[idx_j]
    return trace


# ----------------------------------------------------------------------
# Seed the tracked pair at the field where it was originally identified
# ----------------------------------------------------------------------
ket_i_seed, ket_j_seed, seed_gap = find_active_pair(B0_SEED)
print(f"Seed at B0={B0_SEED} mT: gap={seed_gap:.6e}")

B0_up = np.arange(B0_SEED + B0_STEP, B0_WINDOW_MAX + B0_STEP / 2, B0_STEP)
B0_down = np.arange(B0_SEED - B0_STEP, B0_WINDOW_MIN - B0_STEP / 2, -B0_STEP)

trace_up = track_direction(B0_up, ket_i_seed, ket_j_seed)
trace_down = track_direction(B0_down, ket_i_seed, ket_j_seed)

full_trace = sorted(trace_down + [(B0_SEED, seed_gap)] + trace_up, key=lambda x: x[0])
B0_values = np.array([x[0] for x in full_trace])
gaps = np.array([x[1] for x in full_trace])

min_idx = np.argmin(gaps)
print(f"Minimum gap in window: {gaps[min_idx]:.6e} at B0={B0_values[min_idx]:.4f} mT")
print(f"Gap at window edges: B0={B0_values[0]:.4f} -> {gaps[0]:.6e}, "
      f"B0={B0_values[-1]:.4f} -> {gaps[-1]:.6e}")

if abs(B0_values[min_idx] - B0_SEED) < 5 * B0_STEP and gaps[min_idx] < min(gaps[0], gaps[-1]):
    print("RESULT: genuine local minimum near the seed field -- consistent "
          "with this pair being responsible for the 0.332 mT feature.")
else:
    print("RESULT: no clear local minimum at the seed field -- this pair "
          "does not appear to be the source of the 0.332 mT feature, "
          "despite passing the sector/W checks.")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["B0_mT", "gap"])
    writer.writerows(full_trace)

print(f"Saved to {OUTPUT_FILE}")