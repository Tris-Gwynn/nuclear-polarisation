"""
plot_figure5_two_nucleus_scan.py

Two panels: (a) Delta_Phi_z(B0) for each lambda case, overlaid,
showing how the shared-preparation response evolves as the second
nucleus's coupling grows; (b) max|Delta_Phi_alpha| vs lambda for all
three axes, plus the quadratic-term (midpoint deviation) signature
vs lambda.

Delta_Phi_alpha(B0) = Phi_S(P_alpha=1) - Phi_S(P=0). Named explicitly
as a yield difference, not "c_alpha", since the response here is not
generally linear in P (see the quadratic-term signature).
"""
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_5/figure5_two_nucleus_scan.npz"   # PLACEHOLDER: path if different

data = np.load(INPUT_FILE)
B0_values = data["B0_values"]

LAMBDA_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
CASE_KEYS = [f"lambda_{lam:.2f}" for lam in LAMBDA_VALUES]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel (a): Delta_Phi_z(B0) for each lambda
for case_key in CASE_KEYS:
    p0 = data[f"{case_key}_P0"]
    delta_phi_z = data[f"{case_key}_Pz"] - p0
    axes[0].plot(B0_values, delta_phi_z, label=case_key.replace("_", "="))
axes[0].set_xlabel(r"$B_0$ (mT)")
axes[0].set_ylabel(r"$\Delta\Phi_z$")
axes[0].set_title(r"(a) $\Delta\Phi_z(B_0)$ per $\lambda$ (shared preparation)")
axes[0].legend(fontsize=7)

# Panel (b): max|Delta_Phi_alpha| and midpoint deviation vs lambda
max_dphi_x, max_dphi_y, max_dphi_z, midpoint_dev = [], [], [], []
for lam in LAMBDA_VALUES:
    case_key = f"lambda_{lam:.2f}"
    p0 = data[f"{case_key}_P0"]
    max_dphi_x.append(np.abs(data[f"{case_key}_Px"] - p0).max())
    max_dphi_y.append(np.abs(data[f"{case_key}_Py"] - p0).max())
    max_dphi_z.append(np.abs(data[f"{case_key}_Pz"] - p0).max())
    pz_plus = data[f"{case_key}_Pz"]
    pz_minus = data[f"{case_key}_Pz_minus"]
    midpoint_dev.append(np.abs(p0 - 0.5 * (pz_plus + pz_minus)).max())

axes[1].plot(LAMBDA_VALUES, max_dphi_x, "o-", label=r"$\max|\Delta\Phi_x|$")
axes[1].plot(LAMBDA_VALUES, max_dphi_y, "s-", label=r"$\max|\Delta\Phi_y|$")
axes[1].plot(LAMBDA_VALUES, max_dphi_z, "^-", label=r"$\max|\Delta\Phi_z|$")
axes[1].plot(LAMBDA_VALUES, midpoint_dev, "d--", color="gray",
             label="midpoint deviation\n(quadratic-term signature)")
axes[1].set_xlabel(r"$\lambda$ ($A_2=\lambda A_1$, coaxial)")
axes[1].set_ylabel("response magnitude")
axes[1].set_title(r"(b) response vs coupling strength")
axes[1].legend(fontsize=7)

fig.tight_layout()
fig.savefig("Figure_5/figure5_two_nucleus_scan.png", dpi=300)
print("Saved figure5_two_nucleus_scan.png")