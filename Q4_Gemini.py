"""
Q4: Contrast (Delta Phi_S) at a fixed low-field point vs number of modelled nuclei.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RPM_Experiment_Builder import RPMBuilder

# Note: 8_real_nuc is excluded here for execution speed, but can be safely added 
# to this list if running on a cluster node with sufficient RAM.
CASES = ['toy_1_nuc', 'toy_2_nuc', '4_real_nuc']
LABELS = ['1 Nucleus\n(FAD N5)', '2 Nuclei\n(+TrpH H1)', '4 Nuclei\n(+FAD N10, TrpH N1)']

B0_FIXED = 0.1 # mT
T_MAX = 5.0
contrasts = []

for case in CASES:
    builder = RPMBuilder(case, use_dipolar=False)
    y0 = builder.simulate_yield(B0_FIXED, T_MAX, p_val=0.0)[0]
    y1 = builder.simulate_yield(B0_FIXED, T_MAX, p_val=1.0, pol_axis='z')[0]
    contrasts.append(y1 - y0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(LABELS, contrasts, color=['C0', 'C1', 'C3'], edgecolor='k')
ax.axhline(0, color='k', lw=1)

ax.set_ylabel(rf'Polarisation Contrast $\Delta \Phi_S$ (at $B_0={B0_FIXED}$ mT)')
ax.set_title('Q4: Effect of Hilbert Space Expansion')

plt.tight_layout()
plt.savefig('q4_nuclei_scaling.png', dpi=200)
print("Saved q4_nuclei_scaling.png")