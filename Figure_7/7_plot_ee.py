"""
plot_figure7_offaxis_jd.py

Grouped bar chart: max|c_x|, max|c_y|, max|c_z| per panel (baseline,
exchange, dipolar, both), showing whether J/D suppress, enhance, or
leave untouched each transverse component individually.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_7/figure7_offaxis_jd_data.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)

PANELS = ["a_baseline", "b_exchange", "c_dipolar", "d_both"]
PANEL_LABELS = {
    "a_baseline": "baseline",
    "b_exchange": "exchange",
    "c_dipolar": "dipolar",
    "d_both": "both",
}

max_cx, max_cy, max_cz = [], [], []
for panel_key in PANELS:
    p0 = data[f"{panel_key}_P0"]
    max_cx.append(np.abs(data[f"{panel_key}_Px"] - p0).max())
    max_cy.append(np.abs(data[f"{panel_key}_Py"] - p0).max())
    max_cz.append(np.abs(data[f"{panel_key}_Pz"] - p0).max())

x = np.arange(len(PANELS))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - width, max_cx, width, label=r"$\max|c_x|$")
ax.bar(x,         max_cy, width, label=r"$\max|c_y|$")
ax.bar(x + width, max_cz, width, label=r"$\max|c_z|$")
ax.set_xticks(x)
ax.set_xticklabels([PANEL_LABELS[p] for p in PANELS])
ax.set_ylabel(r"$\max_{B_0}|c_\alpha|$")
ax.legend()

fig.tight_layout()
fig.savefig("Figure_7/figure7_offaxis_jd.png", dpi=300)
print("Saved figure7_offaxis_jd.png")