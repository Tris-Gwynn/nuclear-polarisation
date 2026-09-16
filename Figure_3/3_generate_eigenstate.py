"""
run_figure3_main.py

Generates the main Figure 3 sweep: Phi_S(B0) for Pz = +1, 0, -1,
B parallel z, same hyperfine tensor as Figures 1-2.
"""
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import numpy as np
from tqdm import tqdm
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (must match Figures 1-2)
# ----------------------------------------------------------------------
D_SPINS = [0.5]
A_SPINS = []

A_TENSOR_D = [np.diag([1.0, -0.25, -0.5])]
A_TENSOR_A = []

D_TENSOR = None
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match Figures 1-2
K_T = 1.0                 # PLACEHOLDER: must match Figures 1-2
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

# B parallel z
THETA, PHI = 0.0, 0.0

# ----------------------------------------------------------------------
# B0 sweep (PLACEHOLDER: match the range used in Figures 1-2)
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000

OUTPUT_FILE = "Figure_3/figure3_main_data.npz"

# ----------------------------------------------------------------------
builder = RPMBuilder(
    d_spins=D_SPINS, a_spins=A_SPINS,
    a_tensor_d=A_TENSOR_D, a_tensor_a=A_TENSOR_A,
    d_tensor=D_TENSOR, j_ex=J_EX,
    k_s=K_S, k_t=K_T, k_r=K_R,
    use_ciss=USE_CISS, chi_percent=CHI_PERCENT
)


def phi_s(B0, p_val):
    S_sh, Tp_sh, T0_sh, Tm_sh = builder.yield_(
        B0, p_val=p_val, pol_axis='z', theta=THETA, phi=PHI
    )
    return S_sh


B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)
curve_plus = np.zeros(N_B0)
curve_zero = np.zeros(N_B0)
curve_minus = np.zeros(N_B0)

for i, B0 in enumerate(tqdm(B0_values, desc="Figure 3 main sweep")):
    curve_plus[i] = phi_s(B0, +1.0)
    curve_zero[i] = phi_s(B0, 0.0)
    curve_minus[i] = phi_s(B0, -1.0)

np.savez(
    OUTPUT_FILE,
    B0_values=B0_values,
    curve_plus=curve_plus,
    curve_zero=curve_zero,
    curve_minus=curve_minus,
)
print(f"Saved to {OUTPUT_FILE}")

# Consistency check: Phi_S(0) should equal the mean of Phi_S(+1), Phi_S(-1)
# to numerical precision, given the affine result derived for Figure 2.
midpoint_check = np.abs(curve_zero - 0.5 * (curve_plus + curve_minus))
print(f"Max deviation from exact midpoint: {midpoint_check.max():.3e}")