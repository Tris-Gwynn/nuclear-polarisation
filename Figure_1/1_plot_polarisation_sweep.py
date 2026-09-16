"""
plot_cases.py

Loads case_data.npz (from run_cases.py) and plots Phi_S vs P_z
for Cases 1-3.
"""
import numpy as np
import matplotlib.pyplot as plt

CASE_FILE = "Figure_1/case_data.npz"

cases = np.load(CASE_FILE)
Pz_values = cases["Pz_values"]
case_curves = [
    cases["case_1_curve"],
    cases["case_2_curve"],
    cases["case_3_curve"],
]
case_labels = [
    "Case 1: weak response",
    "Case 2: strong response",
    "Case 3: reversed sign",
]

fig, ax = plt.subplots(figsize=(6, 4.5))
for curve, label in zip(case_curves, case_labels):
    if curve.size > 0:
        ax.plot(Pz_values, curve, marker="o", markersize=3, label=label)
ax.set_xlabel(r"$P_z$")
ax.set_ylabel(r"$\Phi_S$")
ax.set_title("Polarisation response")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("Figure_1/case_curves.png", dpi=300)
print("Saved case_curves.png")