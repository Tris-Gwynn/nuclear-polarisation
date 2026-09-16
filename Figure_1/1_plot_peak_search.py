import numpy as np
import matplotlib.pyplot as plt

SCREENING_FILE = "Figure_1/screening_data.npz"   # PLACEHOLDER: path if different

screening = np.load(SCREENING_FILE)
B0_values = screening["B0_values"]
m_values = screening["m_values"]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(B0_values, m_values, color="black")
ax.axhline(0, color="gray", linewidth=0.6)
ax.set_xlabel(r"$B_0$ (mT)")
ax.set_ylabel(r"$m(B_0)$")
ax.set_title("Screening curve")

fig.tight_layout()
fig.savefig("Figure_1/screening_curve.png", dpi=300)
print("Saved screening_curve.png")