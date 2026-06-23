import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST be before importing pyplot
import matplotlib.pyplot as plt
import qutip as qt

# Drop and rebuild the cached module references if they exist
import sys
import importlib
if 'solver' in sys.modules:
    importlib.reload(sys.modules['solver'])

from core import (NSpinRPMSystem, 
                  get_n_spin_anisotropic_hyperfine, 
                  get_n_spin_zeeman,
                  get_n_spin_dipolar,
                  get_n_spin_exchange)
from solver import NPolarizedSolver
from constants import GYRO_E

# =============================================================================
# 1. EXPERIMENT CONFIGURATION
# =============================================================================

# --- A. Radical Pair & Nuclear Spins ---
# Define spin quantum numbers (e.g., 0.5 for 1H, 1.0 for 14N)
D_SPINS = [0.5] 
A_SPINS = [] 

# Hyperfine Tensors (in MHz)
# Match Mathematica: Ax = 1.0, Ay = 1.0, Az = 0.5
A_X = 1.0 / GYRO_E
A_Y = 1.0 / GYRO_E
A_Z = 0.5 / GYRO_E

# Construct the anisotropic tensor
A_TENSOR_TEST = np.diag([A_X, A_Y, A_Z])

A_TENSOR_D_LIST = [A_TENSOR_TEST] 
A_TENSOR_A_LIST = []

# Electron-Electron Couplings (in MHz)
J_EX = 0.0                      # Isotropic Exchange
D_TENSOR = np.zeros((3, 3))     # Dipolar tensor (explicit 3x3 zero matrix)


# --- B. External Magnetic Fields ---
# Static Field (B0)
B0 = 50.0 / GYRO_E                        # Static field strength (mT)
THETA = 0.0                     # Molecular orientation angle relative to B0
PHI = 0.0                       
print(B0)
# tNMR RF Driving Field (B1)
B1_RF_MT = 0.0                 # RF amplitude in mT (Set to 0.0 to disable RF field)
RF_FREQ_MHZ = 0.0           # 17 kHz converted to MHz
GYRO_1H_MHZ = 0*0.04258 * 2 * np.pi # Proton gyromagnetic ratio


# --- C. Initial State (Nuclear Polarization) ---
# Defined as vectors [Px, Py, Pz]
# E.g., Thermal = [0,0,0] | Longitudinal = [0,0,1] | Transverse = [1,0,0]
P_D_LIST = [[0.0, 0.0, 0.0]]
P_A_LIST = [] 


# --- D. Chemical Kinetics & Time Resolution ---
k_S = 0.0                       # Singlet recombination rate (1/µs)
k_T = 0.0                       # Triplet escape rate (1/µs)
T_MAX = 1000               # µs

times = np.linspace(0, T_MAX, 10000)

# --- 2. SYSTEM SETUP ---
print("Assembling system Hamiltonians...")
sys_rpm = NSpinRPMSystem(d_spins=D_SPINS, a_spins=A_SPINS)
solver = NPolarizedSolver(sys_rpm)

H_hf = get_n_spin_anisotropic_hyperfine(sys_rpm, A_TENSOR_D_LIST, A_TENSOR_A_LIST, theta=0.0, phi=0.0)
H_z = get_n_spin_zeeman(sys_rpm, B0, theta=THETA, phi=PHI)
H_dip = get_n_spin_dipolar(sys_rpm, D_TENSOR, theta=THETA, phi=PHI)
H_ex = get_n_spin_exchange(sys_rpm, J_EX)

# Static background Hamiltonian
H_0 = H_hf + H_z + H_dip + H_ex

# Time-dependent spatial operator: RF field acting on donor nucleus 0 along x-axis
# Note: sys_rpm.ID is an array of dictionaries. We select the first nucleus [0] and its 'x' operator.
H_1 = (B1_RF_MT * GYRO_1H_MHZ) * sys_rpm.ID[0]['x']

# QuTiP time-dependent format: [H_static, [H_drive, 'time_function']]
H_tot_t = [H_0, [H_1, 'cos(w * t)']]

# Define the arguments dictionary required by the string coefficient
args = {'w': RF_FREQ_MHZ * 2 * np.pi}

c_ops = solver.get_collapse_ops(k_S, k_T)
pop_ops = solver.get_population_ops()
e_ops = [pop_ops['S'], pop_ops['Tp'], pop_ops['T0'], pop_ops['Tm']]

# --- 3. RUN TIME EVOLUTION ---
print("Evaluating master equation...")
rho0 = solver.get_initial_rho(P_D_LIST, P_A_LIST)
result = qt.mesolve(H_tot_t, rho0, times, c_ops, e_ops=e_ops, args=args)

pop_S  = np.real(result.expect[0])
pop_Tp = np.real(result.expect[1])
pop_T0 = np.real(result.expect[2])
pop_Tm = np.real(result.expect[3])

# --- 4. PLOTTING ---
fig, ax = plt.subplots(figsize=(8, 10))

ax.plot(times, pop_S,  label=r'Singlet ($S$)', color='#1f77b4', linewidth=2)
#ax.plot(times, pop_Tp, label=r'Triplet ($T_+$)', color='#d62728', linewidth=2)
#ax.plot(times, pop_T0, label=r'Triplet ($T_0$)', color='#2ca02c', linewidth=2)
#ax.plot(times, pop_Tm, label=r'Triplet ($T_-$)', color='#ff7f0e', linewidth=2)

ax.set_xlabel(r'Time ($\mu$s)')
ax.set_ylabel('Population')
ax.set_xlim(0, T_MAX)
ax.set_ylim(-0.02, 1.02)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = 'state_populations_evolution.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Time-resolved population plot successfully saved to {plot_filename}")