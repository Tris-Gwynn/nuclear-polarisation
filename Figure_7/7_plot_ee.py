import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=1.1)
palette = sns.color_palette("bright")

INPUT_FILE = "Figure_7/figure7_offaxis_jd_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

PANELS = ["a_baseline", "b_exchange", "c_dipolar", "d_both"]
PANEL_TITLES = {
    "a_baseline": r"(a) baseline ($J=0$, $D=0$)",
    "b_exchange": r"(b) exchange ($J\neq0$, $D=0$)",
    "c_dipolar":  r"(c) dipolar ($J=0$, $D\neq0$)",
    "d_both":     r"(d) both ($J\neq0$, $D\neq0$)",
}

COMPONENTS = [
    ("Px", r"$\Delta\Phi_x$", palette[3]),
    ("Py", r"$\Delta\Phi_y$", palette[2]),
    ("Pz", r"$\Delta\Phi_z$", palette[0]),
]

fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True)

for ax, panel_key in zip(axes, PANELS):
    p0 = data[f"{panel_key}_P0"]
    for key, label, color in COMPONENTS:
        ax.plot(B0_values, data[f"{panel_key}_{key}"] - p0,
                label=label, color=color, linewidth=1.4)

    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xlim(B0_values.min(), B0_values.max())
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(PANEL_TITLES[panel_key], fontsize=9)

axes[0].set_ylabel(r"$\Delta\Phi_\alpha$")
axes[-1].legend(frameon=False, fontsize=8)

sns.despine(fig=fig)
fig.tight_layout(w_pad=0.8)
fig.savefig("Figure_7/figure7_offaxis_jd.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("Figure_7/figure7_offaxis_jd.pdf", bbox_inches="tight", pad_inches=0.02)
print("Saved figure7_offaxis_jd.png")