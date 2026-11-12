import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=1.1)

INPUT_FILE = "Figure_6/figure6_offaxis_rates.npz" 
data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

AXES = ["Px", "Py", "Pz"]
AXIS_TITLES = {
    "Px": r"$\Delta\Phi_x$",
    "Py": r"$\Delta\Phi_y$",
    "Pz": r"$\Delta\Phi_z$",
}
RATES = ["baseline", "half_triplet", "slow_triplet"]
RATE_LABELS = {
    "baseline": r"$k_T=1.0$",
    "half_triplet": r"$k_T=0.5$",
    "slow_triplet": r"$k_T=0.1$",
}

# Baseline in black, reduced rates on a green ramp (darker = slower)
greens = sns.color_palette("Greens", as_cmap=True)
RATE_COLORS = {
    "baseline": "black",
    "half_triplet": greens(0.5),
    "slow_triplet": greens(0.95),
}
RATE_STYLES = {
    "baseline": "--",
    "half_triplet": "-",
    "slow_triplet": "-",
}

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

for ax, axis_key in zip(axes, AXES):
    for rate in RATES:
        delta_phi = data[f"{rate}_{axis_key}"] - data[f"{rate}_P0"]
        ax.plot(B0_values, delta_phi, label=RATE_LABELS[rate],
                color=RATE_COLORS[rate], linestyle=RATE_STYLES[rate], linewidth=1.4)

    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xlim(B0_values.min(), B0_values.max())
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(AXIS_TITLES[axis_key])

axes[0].set_ylabel(r"$\Delta\Phi_S$")
axes[-1].legend(frameon=False, fontsize=8)

sns.despine(fig=fig)
fig.tight_layout(w_pad=0.8)
fig.savefig("Figure_6/figure6_offaxis_rates.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("Figure_6/figure6_offaxis_rates.pdf", bbox_inches="tight", pad_inches=0.02)
print("Saved figure6_offaxis_rates")