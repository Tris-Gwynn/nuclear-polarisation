"""
plot_figure5.py

Three panels: (a) one nucleus, (b) two nuclei on donor, (c) one donor +
one acceptor nucleus. All tensors A1, all prepared with the same
common Pz. Each shows Phi_S(B0) for Pz = +1, 0, -1.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_5/figure5_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

CASES = ["one_nucleus", "two_on_donor", "donor_acceptor"]
CASE_TITLES = {
    "one_nucleus": "(a) One nucleus",
    "two_on_donor": "(b) Two nuclei, both on donor",
    "donor_acceptor": "(c) One donor + one acceptor nucleus",
}
PREPARATIONS = ["Pz_plus", "P0", "Pz_minus"]
PREPARATION_LABELS = {
    "Pz_plus": r"$P_z=+1$",
    "P0": r"$P_z=0$",
    "Pz_minus": r"$P_z=-1$",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for ax, case_key in zip(axes, CASES):
    for prep in PREPARATIONS:
        ax.plot(B0_values, data[f"{case_key}_{prep}"], label=PREPARATION_LABELS[prep])
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(CASE_TITLES[case_key])

axes[0].set_ylabel(r"$\Phi_S$")
axes[-1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_5/figure5.png", dpi=300)
print("Saved figure5.png")