"""
plot_figure5_two_nucleus_scan.py

Two panels: (a) c_z(B0) for each lambda case, overlaid, showing how
the shared-preparation response evolves as the second nucleus's
coupling grows; (b) max|c_alpha| vs lambda for all three axes, plus
the quadratic-term (midpoint deviation) signature vs lambda.
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_5/figure5_two_nucleus_scan.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

LAMBDA_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
CASE_KEYS = [f"lambda_{lam:.2f}" for lam in LAMBDA_VALUES]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel (a): c_z(B0) for each lambda
for case_key in CASE_KEYS:
    p0 = data[f"{case_key}_P0"]
    cz = data[f"{case_key}_Pz"] - p0
    axes[0].plot(B0_values, cz, label=case_key.replace("_", "="))
axes[0].set_xlabel(r"$B_0$ (mT)")
axes[0].set_ylabel(r"$c_z$")
axes[0].set_title(r"(a) $c_z(B_0)$ per $\lambda$ (shared preparation)")
axes[0].legend(fontsize=7)

# Panel (b): max|c_alpha| and midpoint deviation vs lambda
max_cx, max_cy, max_cz, midpoint_dev = [], [], [], []
for lam in LAMBDA_VALUES:
    case_key = f"lambda_{lam:.2f}"
    p0 = data[f"{case_key}_P0"]
    max_cx.append(np.abs(data[f"{case_key}_Px"] - p0).max())
    max_cy.append(np.abs(data[f"{case_key}_Py"] - p0).max())
    max_cz.append(np.abs(data[f"{case_key}_Pz"] - p0).max())
    pz_plus = data[f"{case_key}_Pz"]
    pz_minus = data[f"{case_key}_Pz_minus"]
    midpoint_dev.append(np.abs(p0 - 0.5 * (pz_plus + pz_minus)).max())

axes[1].plot(LAMBDA_VALUES, max_cx, "o-", label=r"$\max|c_x|$")
axes[1].plot(LAMBDA_VALUES, max_cy, "s-", label=r"$\max|c_y|$")
axes[1].plot(LAMBDA_VALUES, max_cz, "^-", label=r"$\max|c_z|$")
axes[1].plot(LAMBDA_VALUES, midpoint_dev, "d--", color="gray",
             label="midpoint deviation\n(quadratic-term signature)")
axes[1].set_xlabel(r"$\lambda$ ($A_2=\lambda A_1$, coaxial)")
axes[1].set_ylabel("response magnitude")
axes[1].set_title(r"(b) response vs coupling strength")
axes[1].legend(fontsize=7)

fig.tight_layout()
fig.savefig("Figure_5/figure5_two_nucleus_scan.png", dpi=300)
print("Saved figure5_two_nucleus_scan.png")