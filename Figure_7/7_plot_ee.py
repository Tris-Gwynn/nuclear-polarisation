"""
plot_jd_robustness.py

2x2 figure: (a) baseline, (b) +exchange, (c) +dipolar, (d) +both.
Each shows Phi_S(B0) for Pz = +1, 0, -1, one nucleus, B parallel z.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_7/jd_robustness_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

PANELS = ["a_baseline", "b_exchange", "c_dipolar", "d_both"]
PANEL_TITLES = {
    "a_baseline": r"(a) $H_Z+H_{\rm hf}$",
    "b_exchange": r"(b) $H_Z+H_{\rm hf}+H_{\rm ex}$",
    "c_dipolar": r"(c) $H_Z+H_{\rm hf}+H_{\rm dip}$",
    "d_both": r"(d) $H_Z+H_{\rm hf}+H_{\rm ex}+H_{\rm dip}$",
}
PREPARATIONS = ["Pz_plus", "P0", "Pz_minus"]
PREPARATION_LABELS = {
    "Pz_plus": r"$P_z=+1$",
    "P0": r"$P_z=0$",
    "Pz_minus": r"$P_z=-1$",
}

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)

for ax, panel_key in zip(axes.flat, PANELS):
    for prep in PREPARATIONS:
        ax.plot(B0_values, data[f"{panel_key}_{prep}"], label=PREPARATION_LABELS[prep])
    ax.set_title(PANEL_TITLES[panel_key])

for ax in axes[-1, :]:
    ax.set_xlabel(r"$B_0$ (mT)")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Phi_S$")

axes[0, 0].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_7/jd_robustness.png", dpi=300)
print("Saved jd_robustness.png")