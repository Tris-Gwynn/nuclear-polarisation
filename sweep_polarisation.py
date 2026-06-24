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
    return np.real(result.expect[0])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # Radical pair and nuclear spins
    ACTIVE_CASE = '4_real_nuc'
    sys_config = get_nuclear_case(ACTIVE_CASE,
                                  nucleus_location='donor',
                                  anisotropy='isotropic')
    D_SPINS = sys_config['D_SPINS']
    A_SPINS = sys_config['A_SPINS']
    A_TENSOR_D_LIST = sys_config['A_TENSOR_D_LIST']
    A_TENSOR_A_LIST = sys_config['A_TENSOR_A_LIST']

    # Electron-electron coupling
    J_EX = 0.0
    D_TENSOR = np.zeros((3, 3))

    # Static field (magnitude in mT, orientation in rad)
    B0 = 0.05
    THETA = 0.0
    PHI = 0.0

    # Time-dependent RF field. B1_RF_MT = 0 disables it.
    B1_RF_MT = 0.0
    RF_FREQ_MHZ = 0.0
    GYRO_1H_MHZ = 0.04258 * 2 * np.pi

    # Recombination rates. Both zero means unitary dynamics.
    k_S = 1
    k_T = 0.01

    # Time grid
    T_MAX = 5
    TIME_STEPS = 500
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
    sys_rpm = NSpinRPMSystem(d_spins=D_SPINS, a_spins=A_SPINS)
    solver = NPolarizedSolver(sys_rpm)

    H_hf = get_n_spin_anisotropic_hyperfine(sys_rpm, A_TENSOR_D_LIST, A_TENSOR_A_LIST,
                                            theta=0.0, phi=0.0)
    H_z = get_n_spin_zeeman(sys_rpm, B0, theta=THETA, phi=PHI)
    H_dip = get_n_spin_dipolar(sys_rpm, D_TENSOR, theta=THETA, phi=PHI)
    H_ex = get_n_spin_exchange(sys_rpm, J_EX)

    H_0 = H_hf + H_z + H_dip + H_ex
    H_1 = (B1_RF_MT * GYRO_1H_MHZ) * sys_rpm.ID[0]['y']

    H_tot_t = [H_0, [H_1, 'sin(w * t)']] if B1_RF_MT > 0 else H_0
    args = {'w': RF_FREQ_MHZ * 2 * np.pi}

    c_ops = solver.get_collapse_ops(k_S, k_T)
    pop_ops = solver.get_population_ops()
    e_ops = [pop_ops['S']]

    # -------------------------------------------------------------------------
    # Build the two initial states
    # -------------------------------------------------------------------------
    def build_rho0(p_val):
        P_D_LIST = [[0.0, 0.0, p_val] for _ in range(len(D_SPINS))]
        P_A_LIST = [[0.0, 0.0, p_val] for _ in range(len(A_SPINS))]
        return solver.get_initial_rho(P_D_LIST, P_A_LIST)

    rho0_list = [build_rho0(P_LOW), build_rho0(P_HIGH)]

    # -------------------------------------------------------------------------
    # Parallel solve (both runs at once)
    # -------------------------------------------------------------------------
    print(f"Solving P_z = {P_LOW} and P_z = {P_HIGH} across {WORKERS} workers...")
    with mp.Pool(processes=WORKERS,
                 initializer=pool_init,
                 initargs=(H_tot_t, c_ops, e_ops, times, args)) as pool:
        yield_low, yield_high = pool.map(worker_solve, rho0_list)
    print("Execution complete.")

    # -------------------------------------------------------------------------
    # Save raw traces
    # -------------------------------------------------------------------------
    np.save(f'singlet_P0_{ACTIVE_CASE}.npy', yield_low)
    np.save(f'singlet_P1_{ACTIVE_CASE}.npy', yield_high)
    np.save(f'times_{ACTIVE_CASE}.npy', times)
    print(f"Traces saved (singlet_P0/P1_{ACTIVE_CASE}.npy, times_{ACTIVE_CASE}.npy)")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, yield_high, color='C3', lw=1.5,
            label=rf'$P_z = {P_HIGH:.0f}$ (fully polarized)')
    ax.plot(times, yield_low, color='0.2', lw=1.5,
            label=rf'$P_z = {P_LOW:.0f}$ (unpolarized)')
    

    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Singlet Yield ($S$)')
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f'Singlet Yield: Polarized vs. Unpolarized Nuclei\n'
                 f'(B0 = {B0} mT | Case: {ACTIVE_CASE})')
    ax.legend(frameon=False)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    fname = f'singlet_two_runs_{ACTIVE_CASE}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {fname}")