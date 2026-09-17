"""
plot_figure7_offaxis_jd.py

Four panels, one per condition (baseline, exchange, dipolar, both),
each showing Delta_Phi_x, Delta_Phi_y, Delta_Phi_z vs B0 -- same
layout as Figure 6.

Delta_Phi_alpha(B0) = Phi_S(P_alpha=1) - Phi_S(P=0).
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_7/figure7_offaxis_jd_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

PANELS = ["a_baseline", "b_exchange", "c_dipolar", "d_both"]
PANEL_TITLES = {
    "a_baseline": "(a) baseline ($J=0$, $D=0$)",
    "b_exchange": "(b) exchange ($J\\neq0$, $D=0$)",
    "c_dipolar":  "(c) dipolar ($J=0$, $D\\neq0$)",
    "d_both":     "(d) both ($J\\neq0$, $D\\neq0$)",
}

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharey=True)

for ax, panel_key in zip(axes, PANELS):
    p0 = data[f"{panel_key}_P0"]
    ax.plot(B0_values, data[f"{panel_key}_Px"] - p0, label=r"$\Delta\Phi_x$")
    ax.plot(B0_values, data[f"{panel_key}_Py"] - p0, label=r"$\Delta\Phi_y$")
    ax.plot(B0_values, data[f"{panel_key}_Pz"] - p0, label=r"$\Delta\Phi_z$")
    ax.set_xlabel(r"$B_0$ (mT)")
    ax.set_title(PANEL_TITLES[panel_key])

axes[0].set_ylabel(r"$\Delta\Phi_\alpha$")
axes[-1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_7/figure7_offaxis_jd.png", dpi=300)
print("Saved figure7_offaxis_jd.png")