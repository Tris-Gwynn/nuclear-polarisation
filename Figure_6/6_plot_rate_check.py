"""
plot_rate_robustness.py

Two panels: (a) baseline k_S=k_T=1.0, (b) slow triplet k_T=0.5*k_S.
Each shows Phi_S(B0) for Pz = +1, 0, -1, one nucleus, B parallel z.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_6/rate_robustness_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

CASES = ["baseline", "slow_triplet"]
CASE_TITLES = {
    "baseline": r"(a) $k_S=k_T=1.0$",
    "slow_triplet": r"(b) $k_S=1.0,\ k_T=0.1$",
}
PREPARATIONS = ["Pz_plus", "P0", "Pz_minus"]
PREPARATION_LABELS = {
    "Pz_plus": r"$P_z=+1$",
    "P0": r"$P_z=0$",
    "Pz_minus": r"$P_z=-1$",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, case_key in zip(axes, CASES):
    for prep in PREPARATIONS:
        ax.plot(B0_values, data[f"{case_key}_{prep}"], label=PREPARATION_LABELS[prep])
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(CASE_TITLES[case_key])

axes[0].set_ylabel(r"$\Phi_S$")
axes[-1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_6/rate_robustness.png", dpi=300)
print("Saved rate_robustness.png")