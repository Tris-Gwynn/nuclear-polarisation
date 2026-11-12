import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=1.1)
palette = sns.color_palette("bright")

OFFAXIS_FILE = "Figure_4/figure4_offaxis_data.npz"  

offaxis = np.load(OFFAXIS_FILE)
B0_values = offaxis["B0_values"]
p0 = offaxis["P0"]

COMPONENTS = [
    ("Px", r"$\Delta\Phi_x$", palette[3]),
    ("Py", r"$\Delta\Phi_y$", palette[2]),
    ("Pz", r"$\Delta\Phi_z$", palette[0]),
]

fig, ax = plt.subplots(figsize=(10, 3))

for key, label, color in COMPONENTS:
    ax.plot(B0_values, offaxis[key] - p0, label=label, color=color, linewidth=1.4)

ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)

ax.set_xlim(B0_values.min(), B0_values.max())
ax.set_xlabel(r"$B_0$ (mT)")
ax.set_ylabel(r"$\Delta\Phi_\alpha$")
ax.set_title(r"Off-axis field, $(\theta,\phi)=(\pi/4,\pi/4)$")
ax.legend(frameon=False)

sns.despine(fig=fig)
fig.tight_layout()
fig.savefig("Figure_4/figure4.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("Figure_4/figure4.pdf", bbox_inches="tight", pad_inches=0.02)
print("Saved figure4.png")