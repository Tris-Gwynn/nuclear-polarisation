import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=1.1)
palette = sns.color_palette("bright")

INPUT_FILE = "Figure_3/figure3_main_data.npz"
RESONANCE_B0 = [0.1292, 0.3515, 0.3870]

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

CURVES = [
    ("curve_plus",  r"$P_z=+1$", palette[3], "-"),
    ("curve_zero",  r"$P_z=0$",  "0.15",     "--"),
    ("curve_minus", r"$P_z=-1$", palette[4], "-"),
]

fig, ax = plt.subplots(figsize=(10, 3))

for key, label, color, ls in CURVES:
    ax.plot(B0_values, data[key], label=label, color=color,
            linestyle=ls, linewidth=1.4)

for B0_res in RESONANCE_B0:
    ax.axvline(B0_res, color="0.5", linestyle=":", linewidth=0.8, zorder=0)

ax.set_xlim(B0_values.min(), B0_values.max())
ax.set_xlabel(r"$B_0$ (mT)")
ax.set_ylabel(r"$\Phi_S$")
ax.set_title(r"Resonance switching, $B \parallel z$")
ax.legend(frameon=False)

sns.despine(fig=fig)
fig.tight_layout()
fig.savefig("Figure_3/figure3.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("Figure_3/figure3.pdf", bbox_inches="tight", pad_inches=0.02)
print("Saved figure3.png")