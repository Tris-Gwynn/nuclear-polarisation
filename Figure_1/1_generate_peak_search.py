import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (edit as needed)
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
# Screening sweep settings
# ----------------------------------------------------------------------
B0_MIN = 0.0               # PLACEHOLDER: mT
B0_MAX = 1.0              # PLACEHOLDER: mT
N_B0 = 5000                 # PLACEHOLDER: number of points

OUTPUT_FILE = "Figure_1/screening_data.npz"

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


B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)
m_values = np.zeros(N_B0)

for i, B0 in enumerate(tqdm(B0_values, desc="Screening B0")):
    phi_plus = phi_s(B0, +1.0)
    phi_minus = phi_s(B0, -1.0)
    m_values[i] = (phi_plus - phi_minus) / 2.0

np.savez(OUTPUT_FILE, B0_values=B0_values, m_values=m_values)
print(f"Saved to {OUTPUT_FILE}")