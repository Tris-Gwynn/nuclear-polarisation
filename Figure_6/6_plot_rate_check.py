"""
plot_figure6_offaxis_rates.py

Three panels, one per rate case, each showing c_x, c_y, c_z vs B0 at
the off-axis field.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_6/figure6_offaxis_rates.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

CASES = ["baseline", "half_triplet", "slow_triplet"]
CASE_TITLES = {
    "baseline":     r"(a) $k_S=k_T=1.0$",
    "half_triplet": r"(b) $k_S=1.0,\ k_T=0.5$",
    "slow_triplet": r"(c) $k_S=1.0,\ k_T=0.1$",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, case_key in zip(axes, CASES):
    p0 = data[f"{case_key}_P0"]
    ax.plot(B0_values, data[f"{case_key}_Px"] - p0, label=r"$c_x$")
    ax.plot(B0_values, data[f"{case_key}_Py"] - p0, label=r"$c_y$")
    ax.plot(B0_values, data[f"{case_key}_Pz"] - p0, label=r"$c_z$")
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(CASE_TITLES[case_key])

axes[0].set_ylabel(r"$c_\alpha$")
axes[-1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_6/figure6_offaxis_rates.png", dpi=300)
print("Saved figure6_offaxis_rates.png")