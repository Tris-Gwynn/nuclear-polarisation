"""
plot_figure3.py

Loads figure3_main_data.npz and plots Phi_S(B0) for Pz = +1, 0, -1,
with vertical lines marking the three resonance fields.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_3/figure3_main_data.npz"   # PLACEHOLDER: path if different
RESONANCE_B0 = [0.1292, 0.3515, 0.3870]   # must match run_figure3_table.py

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(B0_values, data["curve_plus"], label=r"$P_z=+1$")
ax.plot(B0_values, data["curve_zero"], label=r"$P_z=0$", linestyle="--")
ax.plot(B0_values, data["curve_minus"], label=r"$P_z=-1$")

for B0_res in RESONANCE_B0:
    ax.axvline(B0_res, color="gray", linestyle=":", linewidth=0.8)

ax.set_xlabel(r"$B_0$ (mT)")
ax.set_ylabel(r"$\Phi_S$")
ax.set_title(r"Resonance switching, $B \parallel z$")
ax.legend()

fig.tight_layout()
fig.savefig("Figure_3/figure3.png", dpi=300)
print("Saved figure3.png")