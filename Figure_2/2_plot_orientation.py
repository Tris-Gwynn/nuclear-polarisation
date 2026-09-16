
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_2/figure2_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

ORIENTATIONS = ["Bx", "By", "Bz"]
ORIENTATION_TITLES = {
    "Bx": r"$B \parallel x$",
    "By": r"$B \parallel y$",
    "Bz": r"$B \parallel z$",
}
PREPARATIONS = ["P0", "Px", "Py", "Pz"]
PREPARATION_LABELS = {
    "P0": r"$P=0$",
    "Px": r"$P_x=1$",
    "Py": r"$P_y=1$",
    "Pz": r"$P_z=1$",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, orient in zip(axes, ORIENTATIONS):
    for prep in PREPARATIONS:
        key = f"{orient}_{prep}"
        ax.plot(B0_values, data[key], label=PREPARATION_LABELS[prep])
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(ORIENTATION_TITLES[orient])

axes[0].set_ylabel(r"$\Phi_S$")
axes[-1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_2/figure2.png", dpi=300)
print("Saved figure2.png")