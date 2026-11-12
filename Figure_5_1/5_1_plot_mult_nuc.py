import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=1.1)
palette = sns.color_palette("bright")

INPUT_FILE = "Figure_5_1/figure5_1_two_nucleus_scan_acceptor.npz"   

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

LAMBDA_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
CASE_KEYS = [f"lambda_{lam:.2f}" for lam in LAMBDA_VALUES]


def peak(x):
    """Signed value at the point of largest magnitude."""
    return x[np.argmax(np.abs(x))]


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# lambda = 0 in black, lambda > 0 on a wide blue ramp
blues = sns.color_palette("Blues", as_cmap=True)
ramp = [blues(v) for v in np.linspace(0.35, 1.0, len(LAMBDA_VALUES) - 1)]
lambda_colors = ["black"] + ramp
lambda_styles = ["--"] + ["-"] * (len(LAMBDA_VALUES) - 1)

# Panel (a): Delta_Phi_z(B0) for each lambda
for lam, key, color, ls in zip(LAMBDA_VALUES, CASE_KEYS, lambda_colors, lambda_styles):
    p0 = data[f"{key}_P0"]
    axes[0].plot(B0_values, data[f"{key}_Pz"] - p0,
                 label=rf"$\lambda={lam:.2f}$", color=color, linestyle=ls, linewidth=1.4)

axes[0].axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
axes[0].set_xlim(B0_values.min(), B0_values.max())
axes[0].set_xlabel(r"$B_0$ (mT)")
axes[0].set_ylabel(r"$\Delta\Phi_z$")
axes[0].set_title(r"(a) $\Delta\Phi_z(B_0)$ per $\lambda$ (shared preparation)")
axes[0].legend(frameon=False, fontsize=7)

# Panel (b): signed change at largest magnitude, and deviation from the midpoint relation
dphi_x, dphi_y, dphi_z, midpoint_dev = [], [], [], []
for key in CASE_KEYS:
    p0 = data[f"{key}_P0"]
    dphi_x.append(peak(data[f"{key}_Px"] - p0))
    dphi_y.append(peak(data[f"{key}_Py"] - p0))
    dphi_z.append(peak(data[f"{key}_Pz"] - p0))
    pz_plus = data[f"{key}_Pz"]
    pz_minus = data[f"{key}_Pz_minus"]
    midpoint_dev.append(peak(0.5 * (pz_plus + pz_minus) - p0))

SERIES = [
    (dphi_x,       r"$\Delta\Phi_x$ (largest magnitude)", palette[3], "-"),
    (dphi_y,       r"$\Delta\Phi_y$ (largest magnitude)", palette[2], "-"),
    (dphi_z,       r"$\Delta\Phi_z$ (largest magnitude)", palette[0], "-"),
    (midpoint_dev, "deviation from midpoint relation\n(largest magnitude)", "0.15", "--"),
]
for values, label, color, ls in SERIES:
    axes[1].plot(LAMBDA_VALUES, values, label=label, color=color, linestyle=ls,
                 linewidth=1.4, marker="o", markersize=4, clip_on=False, zorder=3)

axes[1].axhline(0.0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
axes[1].set_xlim(LAMBDA_VALUES[0], LAMBDA_VALUES[-1])
axes[1].set_xlabel(r"$\lambda$ ($A_2=\lambda A_1$, coaxial)")
axes[1].set_ylabel("Signed change at largest magnitude")
axes[1].set_title(r"(b) Response vs coupling strength")
axes[1].legend(frameon=False, fontsize=7, loc="upper left")

sns.despine(fig=fig)
fig.tight_layout()
fig.savefig("Figure_5_1/figure5_1_two_nucleus_scan_acceptor.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("Figure_5_1/figure5_1_two_nucleus_scan_acceptor.pdf", bbox_inches="tight", pad_inches=0.02)
print("Saved figure5_1_two_nucleus_scan_acceptor.png")