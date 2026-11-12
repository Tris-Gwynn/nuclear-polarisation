import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=1.1)
palette = sns.color_palette("bright")

INPUT_FILE = "Figure_2/figure2_data.npz"

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]
RESONANCE_Bx = [0.181436, 0.439488, 0.550710]
RESONANCE_By = [0.116623, 0.328066]
RESONANCE_Bz = [0.129226, 0.35107, 0.387277]

ORIENTATIONS = ["Bx", "By", "Bz"]
ORIENTATION_TITLES = {
    "Bx": r"$B \parallel x$",
    "By": r"$B \parallel y$",
    "Bz": r"$B \parallel z$",
}
ORIENTATION_RESONANCES = {
    "Bx": RESONANCE_Bx,
    "By": RESONANCE_By,
    "Bz": RESONANCE_Bz,
}

COMPONENTS = [
    ("Px", r"$\Delta\Phi_x$", palette[3], +0.0005),
    ("Py", r"$\Delta\Phi_y$", palette[2], 0.0),
    ("Pz", r"$\Delta\Phi_z$", palette[0], -0.0005),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

for ax, orient in zip(axes, ORIENTATIONS):
    p0 = data[f"{orient}_P0"]
    for key, label, color, offset in COMPONENTS:
        ax.plot(B0_values, data[f"{orient}_{key}"] - p0 + offset,
                label=label, color=color, linewidth=1.4)

    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
    for B0_res in ORIENTATION_RESONANCES[orient]:
        ax.axvline(B0_res, color="0.5", linestyle=":", linewidth=0.8, zorder=0)

    ax.set_xlim(B0_values.min(), B0_values.max())
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(ORIENTATION_TITLES[orient])

sns.despine(fig=fig)
axes[0].set_ylabel(r"$\Delta\Phi_\alpha$")
axes[-1].legend(frameon=False, fontsize=8)

fig.tight_layout(w_pad=0.8)
fig.savefig("Figure_2/figure2.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("Figure_2/figure2.pdf", bbox_inches="tight", pad_inches=0.02)
print("Saved figure2.png")