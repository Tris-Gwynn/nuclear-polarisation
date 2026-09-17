"""
plot_figure4.py

Single panel: (theta, phi) = (pi/4, pi/4) (off-axis field). Plots
Delta_Phi_x, Delta_Phi_y, Delta_Phi_z vs B0, instead of the four raw
Phi_S(P) curves.

Delta_Phi_alpha(B0) = Phi_S(P_alpha=1) - Phi_S(P=0).

No data regeneration needed -- P0, Px, Py, Pz are all already saved
in figure4_offaxis_data.npz; this only changes what gets plotted.
"""
import numpy as np
import matplotlib.pyplot as plt

OFFAXIS_FILE = "Figure_4/figure4_offaxis_data.npz"  # PLACEHOLDER: path if different

offaxis = np.load(OFFAXIS_FILE)
B0_values = offaxis["B0_values"]

p0 = offaxis["P0"]
delta_phi_x = offaxis["Px"] - p0
delta_phi_y = offaxis["Py"] - p0
delta_phi_z = offaxis["Pz"] - p0

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(B0_values, delta_phi_x, label=r"$\Delta\Phi_x$")
ax.plot(B0_values, delta_phi_y, label=r"$\Delta\Phi_y$")
ax.plot(B0_values, delta_phi_z, label=r"$\Delta\Phi_z$")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)

ax.set_xlabel(r"$B_0$ (mT)")
ax.set_ylabel(r"$\Delta\Phi_\alpha$")
ax.set_title(r"Off-axis field, $(\theta,\phi)=(\pi/4,\pi/4)$")
ax.legend()

fig.tight_layout()
fig.savefig("Figure_4/figure4.png", dpi=300)
print("Saved figure4.png")