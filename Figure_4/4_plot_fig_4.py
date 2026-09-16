"""
plot_figure4.py

Single panel: (theta, phi) = (pi/4, pi/4) (off-axis field). Plots
Phi_S(B0) for P0, Px, Py, Pz. Figure 2 already covers the aligned
(B parallel z) reference case, so it is not repeated here.
"""
import numpy as np
import matplotlib.pyplot as plt

OFFAXIS_FILE = "Figure_4/figure4_offaxis_data.npz"  # PLACEHOLDER: path if different

offaxis = np.load(OFFAXIS_FILE)
B0_values = offaxis["B0_values"]

PREPARATIONS = ["P0", "Px", "Py", "Pz"]
PREPARATION_LABELS = {
    "P0": r"$P=0$",
    "Px": r"$P_x=1$",
    "Py": r"$P_y=1$",
    "Pz": r"$P_z=1$",
}

fig, ax = plt.subplots(figsize=(7, 5))
for prep in PREPARATIONS:
    ax.plot(B0_values, offaxis[prep], label=PREPARATION_LABELS[prep])

ax.set_xlabel(r"$B_0$ (mT)")
ax.set_ylabel(r"$\Phi_S$")
ax.set_title(r"Off-axis field, $(\theta,\phi)=(\pi/4,\pi/4)$")
ax.legend()

fig.tight_layout()
fig.savefig("Figure_4/figure4.png", dpi=300)
print("Saved figure4.png")