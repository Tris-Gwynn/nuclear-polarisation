"""
Q2: Contrast (Delta Phi_S) vs B0 for varying magnitudes of the hyperfine interaction.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RPM_Experiment_Builder import RPMBuilder
from cases import get_nuclear_case

ACTIVE_CASE = 'toy_1_nuc'
T_MAX = 5.0
B0_ARRAY = np.logspace(-3, 1, 100)
HF_SCALES = [0.1, 1.0, 5.0]

def worker(B0):
    cfg = get_nuclear_case(ACTIVE_CASE)
    contrasts = []
    
    for scale in HF_SCALES:
        # Scale the raw tensors 
        scaled_d = [t * scale for t in cfg['A_TENSOR_D_LIST']]
        scaled_a = [t * scale for t in cfg['A_TENSOR_A_LIST']]
        
        # Override the builder's default tensors
        builder = RPMBuilder(ACTIVE_CASE, a_tensor_d=scaled_d, a_tensor_a=scaled_a, use_dipolar=False)
        
        y0 = builder.simulate_yield(B0, T_MAX, p_val=0.0)[0]
        y1 = builder.simulate_yield(B0, T_MAX, p_val=1.0, pol_axis='z')[0]
        contrasts.append(y1 - y0)
        
    return [B0] + contrasts

if __name__ == '__main__':
    with mp.Pool(processes=25) as pool:
        results = np.array(pool.map(worker, B0_ARRAY))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['C0', 'C1', 'C3']
    
    for i, scale in enumerate(HF_SCALES):
        ax.plot(results[:, 0], results[:, i+1], color=colors[i], lw=2, label=f'{scale}x HF Strength')
        
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax.set_ylabel(r'Polarisation Contrast $\Delta \Phi_S$')
    ax.set_title('Q2: Hyperfine Magnitude Scaling')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('q2_hyperfine_scaling.png', dpi=200)
    print("Saved q2_hyperfine_scaling.png")