"""
Q1: Coherent dynamics (k=0) comparing unpolarised, x-polarised, and z-polarised 
initial states under isotropic vs. anisotropic hyperfine interactions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RPM_Experiment_Builder import RPMBuilder

# Configuration
ACTIVE_CASE = 'toy_1_nuc'
B0 = 0.05  # mT
T_MAX = 0.5
TIMES = np.linspace(0, T_MAX, 1000)

# Build strictly coherent systems
builder_iso = RPMBuilder(ACTIVE_CASE, isotropic_hf=True, use_dipolar=False, k_s=0, k_t=0)
builder_aniso = RPMBuilder(ACTIVE_CASE, isotropic_hf=False, use_dipolar=False, k_s=0, k_t=0)

configs = [
    (1.0, 'z', 'Z-Polarised ($P_z=1$)', 'C3'),
    (1.0, 'x', 'X-Polarised ($P_x=1$)', 'C0'),
    (0.0, 'z', 'Unpolarised ($P=0$)', 'k')
    
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Left Panel: Isotropic, B0 along z (Theta=0)
for p_val, axis, label, color in configs:
    res = builder_iso.simulate_dynamics(B0, TIMES, p_val=p_val, pol_axis=axis)
    ax1.plot(TIMES, res[0], color=color, lw=1.5, label=label) # res[0] is Active Singlet S

ax1.set_title(r'Isotropic HF ($\theta=0$)')
ax1.set_xlabel(r'Time ($\mu$s)')
ax1.set_ylabel(r'Active Singlet Probability $P_S(t)$')
ax1.set_ylim(-0.05, 1.05)
ax1.legend(frameon=False)

# Right Panel: Anisotropic, B0 tilted (Theta=pi/4) to expose off-diagonal terms
for p_val, axis, label, color in configs:
    res = builder_aniso.simulate_dynamics(B0, TIMES, p_val=p_val, pol_axis=axis, theta=np.pi/4)
    ax2.plot(TIMES, res[0], color=color, lw=1.5, label=label)

ax2.set_title(r'Anisotropic HF ($\theta=\pi/4$)')
ax2.set_xlabel(r'Time ($\mu$s)')

plt.tight_layout()
plt.savefig('q1_coherent_dynamics.png', dpi=200)
print("Saved q1_coherent_dynamics.png")