"""
Computes the Magnetically Altered Reaction Yield (MARY) curve for 
polarized vs unpolarized nuclei across a logarithmic B0 sweep.
"""
import time
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RPM_Experiment_Builder import RPMBuilder

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
ACTIVE_CASE = 'toy_1_nuc'
USE_DIPOLAR = False
J_EX = 0.0
THETA = np.pi/2
PHI = np.pi/2

T_MAX = 5.0
P_LOW = 0.0
P_HIGH = 1.0
pol_axis='z'

# B0 Sweep Parameters (Logarithmic from 1 microTesla to 100 milliTesla)
B0_MIN = 1e-3  
B0_MAX = 10.0
B0_POINTS = 100
B0_ARRAY = np.logspace(np.log10(B0_MIN), np.log10(B0_MAX), B0_POINTS)

def compute_mary_point(B0):
    """Worker function to compute final singlet yield for a given B0."""
    # The builder handles all constants, core initialization, and tensor mapping
    builder = RPMBuilder(ACTIVE_CASE, use_dipolar=USE_DIPOLAR, j_ex=J_EX)
    
    # Unpolarized run
    yields_low = builder.simulate_yield(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis=pol_axis, theta=THETA, phi=PHI)
    
    # Polarized run
    yields_x = builder.simulate_yield(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=THETA, phi=PHI)
    yields_y = builder.simulate_yield(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=THETA, phi=PHI)
    yields_z = builder.simulate_yield(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=THETA, phi=PHI)

    return B0, yields_low[0], yields_x[0], yields_y[0], yields_z[0]  # Index 0 corresponds to S_sh (Singlet Yield)

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(f"Sweeping {B0_POINTS} magnetic field points...")

    # Execute CPU parallel sweep
    cores = 25
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_point, B0_ARRAY)

    # Unpack results
    results = np.array(results)
    B0_vals = results[:, 0]
    Yields_low = results[:, 1]
    Yields_x = results[:, 2]
    Yields_y = results[:, 3]
    Yields_z = results[:, 4]

    # -------------------------------------------------------------------------
    # Plotting Dual-Axis MARY Curve
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis: Singlet Yields
    color_low = 'C3' # Red
    color_x = 'C0' # Blue
    color_y = 'C1' # Orange
    color_z = 'C2' # Green

    ax1.plot(B0_vals, Yields_low, color=color_low, lw=2, label=rf'Unpolarized ($P={P_LOW:.0f}$)')
    ax1.plot(B0_vals, Yields_x, color=color_x, lw=2, label=rf'Fully Polarized ($P_x={P_HIGH:.0f}$)')
    ax1.plot(B0_vals, Yields_y, color=color_y, lw=2, label=rf'Fully Polarized ($P_y={P_HIGH:.0f}$)')
    ax1.plot(B0_vals, Yields_z, color=color_z, lw=2, label=rf'Fully Polarized ($P_z={P_HIGH:.0f}$)')

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax1.set_xlim(B0_MIN, B0_MAX)
    ax1.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left', frameon=False)

    
    plt.title('Magnetically Altered Reaction Yield (MARY)')
    plt.tight_layout()
    
    fname = f'mary_sweep_{ACTIVE_CASE}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()

    end_time = time.perf_counter()
    print(f"Sweep complete in {end_time - start_time:.2f} seconds. Saved to {fname}")