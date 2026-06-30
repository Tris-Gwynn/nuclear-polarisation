"""
Q5: Dynamics across Coherent (k=0), Symmetric Open (k_S=k_T), and Asymmetric Open (k_S!=k_T) regimes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RPM_Experiment_Builder import RPMBuilder

ACTIVE_CASE = 'toy_2_nuc'
B0 = 0.1
T_MAX = 5.0
TIMES = np.linspace(0, T_MAX, 500)

# (k_S, k_T, title)
regimes = [
    (0.0, 0.0, r'Coherent ($k_S=k_T=0$)'),
    (1.0, 1.0, r'Symmetric Open ($k_S=k_T=1$)'),
    (1.0, 0.0, r'Asymmetric Open ($k_S=1, k_T=0$)')
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (ks, kt, title) in zip(axes, regimes):
    builder = RPMBuilder(ACTIVE_CASE, k_s=ks, k_t=kt, use_dipolar=False)
    
    res_0 = builder.simulate_dynamics(B0, TIMES, p_val=0.0)
    res_1 = builder.simulate_dynamics(B0, TIMES, p_val=1.0, pol_axis='z')
    
    # For coherent, we plot Active S (index 0). For open, we plot Yield S (index 4).
    target_idx = 0 if (ks == 0 and kt == 0) else 4
    ylabel = r'Active Probability $P_S(t)$' if (ks == 0 and kt == 0) else r'Cumulative Yield $\Phi_S(t)$'
    
    ax.plot(TIMES, res_0[target_idx], 'k-', lw=1.5, label='Unpolarised')
    ax.plot(TIMES, res_1[target_idx], 'C3--', lw=1.5, label='Z-Polarised')
    
    ax.set_title(title)
    ax.set_xlabel(r'Time ($\mu$s)')
    if ax == axes[0]:
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('q5_kinetic_regimes.png', dpi=200)
print("Saved q5_kinetic_regimes.png")