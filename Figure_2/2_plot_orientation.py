"""
plot_figure2_orientation.py

Loads figure2_data.npz and plots Delta_Phi_x, Delta_Phi_y, Delta_Phi_z
vs B0 for each field orientation (B || x, y, z), instead of the four
raw Phi_S(P) curves.

Delta_Phi_alpha(B0) = Phi_S(P_alpha=1) - Phi_S(P=0).

No data regeneration needed -- P0, Px, Py, Pz are all already saved
in figure2_data.npz; this only changes what gets plotted.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_2/figure2_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

ORIENTATIONS = ["Bx", "By", "Bz"]
ORIENTATION_TITLES = {
    "Bx": r"$B \parallel x$",
    "By": r"$B \parallel y$",
    "Bz": r"$B \parallel z$",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, orient in zip(axes, ORIENTATIONS):
    p0 = data[f"{orient}_P0"]
    ax.plot(B0_values, data[f"{orient}_Px"] - p0, label=r"$\Delta\Phi_x$")
    ax.plot(B0_values, data[f"{orient}_Py"] - p0, label=r"$\Delta\Phi_y$")
    ax.plot(B0_values, data[f"{orient}_Pz"] - p0, label=r"$\Delta\Phi_z$")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(ORIENTATION_TITLES[orient])

axes[0].set_ylabel(r"$\Delta\Phi_\alpha$")
axes[-1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_2/figure2.png", dpi=300)
print("Saved figure2.png")