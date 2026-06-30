"""
Q6: Contrast (Delta Phi_S) vs B0 tracking the structural energy shifts introduced by Exchange (J) and Dipolar (D) tensors.
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
J_VAL = 1.5 # 0.5 mT exchange splitting

def worker(B0):
    b_none = RPMBuilder(ACTIVE_CASE, use_dipolar=False, j_ex=0.0)
    b_j    = RPMBuilder(ACTIVE_CASE, use_dipolar=False, j_ex=J_VAL)
    b_d    = RPMBuilder(ACTIVE_CASE, use_dipolar=True,  j_ex=0.0)
    b_both = RPMBuilder(ACTIVE_CASE, use_dipolar=True,  j_ex=J_VAL)
    
    builders = [b_none, b_j, b_d, b_both]
    contrasts = []
    
    for b in builders:
        y0 = b.simulate_yield(B0, T_MAX, p_val=0.0)[0]
        y1 = b.simulate_yield(B0, T_MAX, p_val=1.0, pol_axis='z')[0]
        contrasts.append(y1 - y0)
        
    return [B0] + contrasts

if __name__ == '__main__':
    with mp.Pool(processes=25) as pool:
        results = np.array(pool.map(worker, B0_ARRAY))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results[:, 0], results[:, 1], 'k-',  lw=2, label='No Coupling')
    ax.plot(results[:, 0], results[:, 2], 'C0--', lw=2, label=rf'Exchange Only ($J={J_VAL}$)')
    ax.plot(results[:, 0], results[:, 3], 'C1-.', lw=2, label='Dipolar Only')
    ax.plot(results[:, 0], results[:, 4], 'C3:',  lw=2, label='Exchange + Dipolar')
        
    ax.axhline(0, color='gray', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax.set_ylabel(r'Polarisation Contrast $\Delta \Phi_S$')
    ax.set_title('Q6: Effect of e-e Couplings on LCR Placements')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('q6_exchange_and_dipolar.png', dpi=200)
    print("Saved q6_exchange_and_dipolar.png")