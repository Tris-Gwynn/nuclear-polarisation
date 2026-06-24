"""
Singlet yield for two nuclear polarization extremes, run in parallel.

Solves the unpolarized case (P_z = 0) and the fully polarized case (P_z = 1)
concurrently in two worker processes, then plots both yield traces on a single
axis. Two tasks means two workers is the most that can help.
"""

import os
# QuTiP's sparse integrator is single-threaded, so BLAS threads add nothing.
# Pinning to 1 also stops the two workers oversubscribing cores during setup.
os.environ['OMP_NUM_THREADS'] = '1'

import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import qutip as qt
from constants import (GYRO_E, GYRO_H1, GYRO_N14)
from core import (NSpinRPMSystem,
                  get_n_spin_anisotropic_hyperfine,
                  get_n_spin_zeeman,
                  get_n_spin_dipolar,
                  get_n_spin_exchange)
from solver import NPolarizedSolver
from cases import get_nuclear_case


# =============================================================================
# WORKER PROCESS SETUP
# =============================================================================
# The Hamiltonian, operators, time grid and args are identical for both runs.
# They are loaded once per worker via the initializer rather than shipped with
# each task. This is correct under both 'fork' and 'spawn' start methods.

def pool_init(H, c_ops, e_ops, times, args):
    global _H, _c_ops, _e_ops, _times, _args
    _H = H
    _c_ops = c_ops
    _e_ops = e_ops
    _times = times
    _args = args


def worker_solve(rho0):
    """Solve for one initial state. Returns the 1D singlet-yield trace only."""
    result = qt.mesolve(_H, rho0, _times, _c_ops, e_ops=_e_ops, args=_args)
    return [np.real(ex) for ex in result.expect]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # Radical pair and nuclear spins
    ACTIVE_CASE = 'toy_2_nuc'  # 'toy_1_nuc', 'toy_2_nuc', '4_real_nuc'
    sys_config = get_nuclear_case(ACTIVE_CASE,
                                  nucleus_location='donor',
                                  anisotropy='isotropic')
    D_SPINS = sys_config['D_SPINS']
    A_SPINS = sys_config['A_SPINS']
    A_TENSOR_D_LIST = sys_config['A_TENSOR_D_LIST']
    A_TENSOR_A_LIST = sys_config['A_TENSOR_A_LIST']

    # Electron-electron coupling
    J_EX = 0.0
    USE_DIPOLAR = True  # Toggle to switch dipolar coupling on or off

    if USE_DIPOLAR:
        D_TENSOR = np.array([
            [ 0.030687, -0.338269,  0.136643], 
            [-0.338269, -0.218231,  0.195874], 
            [ 0.136643,  0.195874,  0.187544]
        ])
    else:
        D_TENSOR = np.zeros((3, 3))
    
    # Static field (magnitude in mT, orientation in rad)
    B0 = 0.05
    THETA = 0.0 * np.pi / 180 
    PHI = 0.0 * 2 * np.pi /180

    # Time-dependent RF field. B1_RF_MT = 0 disables it.
    B1_RF_MT = 0.0
    RF_FREQ_MHZ = 0.0
    ax = 'x'  # RF field along ' '-axis

    # Recombination rates. Both zero means unitary dynamics.
    k_S = 1.0
    k_T = 0.01

    # --- ADDED: CISS Settings ---
    USE_CISS = False          # True = 9D space with backscatter, False = 8D space
    k_R = 0.0
    CHI_PERCENT = 100                # Backscatter rate
    CHI_INIT = CHI_PERCENT/100 * np.pi / 2     # Chiral phase angle
    CHI_RECOMB = CHI_INIT  # Chiral phase angle for recombination   

    # Time grid
    T_MAX = 25
    TIME_STEPS = 5000
    times = np.linspace(0, T_MAX, TIME_STEPS)

    # The two polarization endpoints to compare
    P_LOW = 0.0
    P_HIGH = 1.0

    # Two tasks, so two workers is the ceiling.
    WORKERS = 2

    # -------------------------------------------------------------------------
    # Static Hamiltonian assembly
    # -------------------------------------------------------------------------
    print("Assembling static system Hamiltonians...")
    sys_rpm = NSpinRPMSystem(d_spins=D_SPINS, a_spins=A_SPINS, use_ciss=USE_CISS)
    solver = NPolarizedSolver(sys_rpm)

    H_hf = get_n_spin_anisotropic_hyperfine(sys_rpm, A_TENSOR_D_LIST, A_TENSOR_A_LIST,
                                            theta=0.0, phi=0.0)
    H_z = get_n_spin_zeeman(sys_rpm, B0, theta=THETA, phi=PHI)
    H_dip = get_n_spin_dipolar(sys_rpm, D_TENSOR, theta=THETA, phi=PHI)
    H_ex = get_n_spin_exchange(sys_rpm, J_EX)

    H_0 = H_hf + H_z + H_dip + H_ex

    if B1_RF_MT > 0:
        # 1. Apply to both electrons
        H_1 = (B1_RF_MT * GYRO_E) * (sys_rpm.SD[ax] + sys_rpm.SA[ax])

        # 2. Apply to all donor nuclei, mapping gamma based on spin magnitude
        for spin_val, nuc in zip(sys_rpm.d_spins, sys_rpm.ID):
            gamma_n = GYRO_H1 if spin_val == 0.5 else GYRO_N14
            H_1 += (B1_RF_MT * gamma_n) * nuc[ax]

        # 3. Apply to all acceptor nuclei
        for spin_val, nuc in zip(sys_rpm.a_spins, sys_rpm.IA):
            gamma_n = GYRO_H1 if spin_val == 0.5 else GYRO_N14
            H_1 += (B1_RF_MT * gamma_n) * nuc[ax]
    else:
        H_1 = 0.0

    H_tot_t = [H_0, [H_1, 'cos(w * t)']] if B1_RF_MT > 0 else H_0
    args = {'w': RF_FREQ_MHZ * 2 * np.pi}

    c_ops = solver.get_collapse_ops(k_S, k_T, kr=k_R, chi=CHI_RECOMB)
    pop_ops = solver.get_population_ops()
    e_ops = [pop_ops['S'],pop_ops['Tp'],pop_ops['T0'],pop_ops['Tm']]

    # -------------------------------------------------------------------------
    # Build the two initial states
    # -------------------------------------------------------------------------
    def build_rho0(p_val, axis='z'):
        axis_map = {
            'x': [p_val, 0.0, 0.0],
            'y': [0.0, p_val, 0.0],
            'z': [0.0, 0.0, p_val]
        }
        p_vec = axis_map[axis]
        P_D_LIST = [p_vec for _ in range(len(D_SPINS))]
        P_A_LIST = [p_vec for _ in range(len(A_SPINS))]
        return solver.get_initial_rho(P_D_LIST, P_A_LIST, chi_init=CHI_INIT)

    axis = 'x'  # Polarization along the z-axis
    rho0_list = [build_rho0(P_LOW, axis), build_rho0(P_HIGH, axis)]

    # -------------------------------------------------------------------------
    # Parallel solve (both runs at once)
    # -------------------------------------------------------------------------
    print(f"Solving P_z = {P_LOW} and P_z = {P_HIGH} across {WORKERS} workers...")
    with mp.Pool(processes=WORKERS,
                 initializer=pool_init,
                 initargs=(H_tot_t, c_ops, e_ops, times, args)) as pool:
        yield_low_all, yield_high_all = pool.map(worker_solve, rho0_list)
    print("Execution complete.")

    # Unpack the traces
    S_low, Tp_low, T0_low, Tm_low = yield_low_all
    S_high, Tp_high, T0_high, Tm_high = yield_high_all

    # -------------------------------------------------------------------------
    # Save raw traces
    # -------------------------------------------------------------------------
    np.save(f'singlet_P0_{ACTIVE_CASE}.npy', yield_low_all)
    np.save(f'singlet_P1_{ACTIVE_CASE}.npy', yield_high_all)
    np.save(f'times_{ACTIVE_CASE}.npy', times)
    print(f"Traces saved (singlet_P0/P1_{ACTIVE_CASE}.npy, times_{ACTIVE_CASE}.npy)")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Colors for specific states
    c_S = 'C3'  # Blue
    c_Tp = 'C2' # Red
    c_T0 = 'C0' # Green
    c_Tm = 'C1' # Purple

    # Top Panel: Unpolarized
    ax1.plot(times, S_low, color=c_S, lw=1.5, label='Singlet ($S$)')
    ax1.plot(times, Tp_low, color=c_Tp, lw=1.5, label='Triplet ($T_+$)')
    ax1.plot(times, T0_low, color=c_T0, lw=1.5, label='Triplet ($T_0$)')
    ax1.plot(times, Tm_low, color=c_Tm, lw=1.5, label='Triplet ($T_-$)')
    
    ax1.set_ylabel('State Population')
    ax1.set_ylim(-0.1, 1.0)
    ax1.set_title(rf'State Evolution: Unpolarized ($P_z = {P_LOW:.0f}$)')
    ax1.legend(loc='upper right', frameon=False, ncol=2)

    # Bottom Panel: Polarized
    ax2.plot(times, S_high, color=c_S, lw=1.5)
    ax2.plot(times, Tp_high, color=c_Tp, lw=1.5)
    ax2.plot(times, T0_high, color=c_T0, lw=1.5)
    ax2.plot(times, Tm_high, color=c_Tm, lw=1.5)
    
    ax2.set_xlabel(r'Time ($\mu$s)')
    ax2.set_ylabel('State Population')
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_title(rf'State Evolution: Fully Polarized ($P_z = {P_HIGH:.0f}$)')

    plt.tight_layout()
    fname = f'populations_two_runs_{ACTIVE_CASE}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {fname}")