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
ACTIVE_CASE = 'toy_2_nuc'
USE_DIPOLAR = False
J_EX = 0.0
THETA = 0.0
PHI = 0.0

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
    yields_high = builder.simulate_yield(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis=pol_axis, theta=THETA, phi=PHI)

    return B0, yields_low[0], yields_high[0]  # Index 0 corresponds to S_sh (Singlet Yield)

if __name__ == '__main__':
    start_time = time.perf_counter()
    print(f"Sweeping {B0_POINTS} magnetic field points...")

    # Execute CPU parallel sweep
    cores = mp.cpu_count()
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_point, B0_ARRAY)

    # Unpack results
    results = np.array(results)
    B0_vals = results[:, 0]
    Yields_low = results[:, 1]
    Yields_high = results[:, 2]
    Contrast = Yields_high - Yields_low

    # -------------------------------------------------------------------------
    # Plotting Dual-Axis MARY Curve
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis: Singlet Yields
    color_low = 'C3' # Red
    color_high = 'C0' # Blue
    ax1.plot(B0_vals, Yields_low, color=color_low, lw=2, label=rf'Unpolarized ($P_z={P_LOW:.0f}$)')
    ax1.plot(B0_vals, Yields_high, color=color_high, lw=2, label=rf'Fully Polarized ($P_z={P_HIGH:.0f}$)')
    
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax1.set_xlim(B0_MIN, B0_MAX)
    ax1.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left', frameon=False)

    # Secondary Axis: Polarization Contrast
    ax2 = ax1.twinx()
    color_contrast = 'k' # Black
    ax2.plot(B0_vals, Contrast, color=color_contrast, lw=2, linestyle='--', label=r'Contrast ($\Delta \Phi_S$)')
    ax2.set_ylabel(r'Polarization Contrast ($\Delta \Phi_S$)', color=color_contrast)
    ax2.tick_params(axis='y', labelcolor=color_contrast)
    ax2.axhline(0, color='gray', linestyle=':', lw=1)
    ax2.legend(loc='center right', frameon=False)

    plt.title('Magnetically Altered Reaction Yield (MARY)')
    plt.tight_layout()
    
    fname = f'mary_sweep_{ACTIVE_CASE}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()

    end_time = time.perf_counter()
    print(f"Sweep complete in {end_time - start_time:.2f} seconds. Saved to {fname}")