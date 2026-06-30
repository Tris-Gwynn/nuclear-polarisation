"""
Q3: Contrast (Delta Phi_S) vs B0 comparing Isotropic and Anisotropic Hyperfine tensors.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RPM_Experiment_Builder import RPMBuilder

ACTIVE_CASE = 'toy_1_nuc'
T_MAX = 5.0
B0_ARRAY = np.logspace(-3, 1, 100)
THETA = 0  # Tilt field to expose anisotropy differences

def worker(B0):
    b_iso = RPMBuilder(ACTIVE_CASE, isotropic_hf=True, use_dipolar=False)
    b_ani = RPMBuilder(ACTIVE_CASE, isotropic_hf=False, use_dipolar=False)
    
    y0_iso = b_iso.simulate_yield(B0, T_MAX, p_val=0.0, theta=THETA)[0]
    y1_iso = b_iso.simulate_yield(B0, T_MAX, p_val=1.0, pol_axis='z', theta=THETA)[0]
    
    y0_ani = b_ani.simulate_yield(B0, T_MAX, p_val=0.0, theta=THETA)[0]
    y1_ani = b_ani.simulate_yield(B0, T_MAX, p_val=1.0, pol_axis='z', theta=THETA)[0]
    
    return B0, (y1_iso - y0_iso), (y1_ani - y0_ani)

if __name__ == '__main__':
    with mp.Pool(processes=25) as pool:
        results = np.array(pool.map(worker, B0_ARRAY))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results[:, 0], results[:, 1], 'C0-', lw=2, label='Isotropic HF')
    ax.plot(results[:, 0], results[:, 2], 'C3--', lw=2, label='Anisotropic HF')
        
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax.set_ylabel(r'Polarisation Contrast $\Delta \Phi_S$')
    ax.set_title(r'Q3: Tensor Symmetry Effect ($\theta=\pi/4$)')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('q3_isotropic_vs_anisotropic.png', dpi=200)
    print("Saved q3_isotropic_vs_anisotropic.png")