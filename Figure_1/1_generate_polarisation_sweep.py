import numpy as np
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (must match run_screening.py)
# ----------------------------------------------------------------------
D_SPINS = [0.5]          # PLACEHOLDER: donor nuclear spins
A_SPINS = []              # PLACEHOLDER: acceptor nuclear spins

A_TENSOR_D = [np.diag([1.0, -0.25, -0.5])]   # mT, one 3x3 tensor per donor nucleus
A_TENSOR_A = []                               # mT, one 3x3 tensor per acceptor nucleus

D_TENSOR = None           # PLACEHOLDER: 3x3 dipolar tensor (mT), or None
J_EX = 0.0                # PLACEHOLDER: exchange coupling (mT)

K_S = 1.0                 # PLACEHOLDER: singlet reaction rate (MHz)
K_T = 1.0                 # PLACEHOLDER: triplet reaction rate (MHz)
K_R = 0.0                 # PLACEHOLDER: CISS backscatter rate (MHz)

USE_CISS = False          # PLACEHOLDER
CHI_PERCENT = 0.0         # PLACEHOLDER

POL_AXIS = 'z'

# ----------------------------------------------------------------------
# Chosen B0 values (fill in after inspecting screening_data.npz)
# ----------------------------------------------------------------------
B0_CASE_1 = 0.05   # PLACEHOLDER: off-resonant / weak-response B0 (mT)
B0_CASE_2 = 0.38728   # PLACEHOLDER: strong-response B0 (mT)
B0_CASE_3 = 0.12923   # PLACEHOLDER: opposite-sign response B0 (mT), or None if absent

N_PZ = 21          # PLACEHOLDER: number of P_z points in [-1, 1]

OUTPUT_FILE = "Figure_1/case_data.npz"

# ----------------------------------------------------------------------
builder = RPMBuilder(
    d_spins=D_SPINS, a_spins=A_SPINS,
    a_tensor_d=A_TENSOR_D, a_tensor_a=A_TENSOR_A,
    d_tensor=D_TENSOR, j_ex=J_EX,
    k_s=K_S, k_t=K_T, k_r=K_R,
    use_ciss=USE_CISS, chi_percent=CHI_PERCENT
)


def phi_s(B0, p_val):
    """Singlet shelved yield Phi_S at given B0 and P_z."""
    S_sh, Tp_sh, T0_sh, Tm_sh = builder.yield_(B0, p_val=p_val, pol_axis=POL_AXIS)
    return S_sh


Pz_values = np.linspace(-1.0, 1.0, N_PZ)
case_B0_values = [B0_CASE_1, B0_CASE_2, B0_CASE_3]
case_curves = []

for B0_case in case_B0_values:
    if B0_case is None:
        case_curves.append(np.array([]))
        continue
    curve = np.array([phi_s(B0_case, p) for p in Pz_values])
    case_curves.append(curve)

np.savez(
    OUTPUT_FILE,
    Pz_values=Pz_values,
    case_B0_values=np.array([np.nan if b is None else b for b in case_B0_values]),
    case_1_curve=case_curves[0],
    case_2_curve=case_curves[1],
    case_3_curve=case_curves[2],
)
print(f"Saved to {OUTPUT_FILE}")