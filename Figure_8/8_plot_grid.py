"""
Plot the orientation map from 8_generate_orientation_map.py.

Three panels (polarisation along x, y, z), each showing a field-integrated
measure of |Delta Phi_alpha| as a function of field direction (theta, phi) in
the hyperfine principal-axis frame. All panels share one colour scale so the
three polarisation directions can be compared directly.

QUANTITY = "area_abs" : integral over 0-1 mT of |Delta Phi_alpha| (mT)
QUANTITY = "max_abs"  : max over 0-1 mT of |Delta Phi_alpha|
"""

import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_8/figure8_orientation_map.npz"
QUANTITY = "area_abs"
OUTPUT_STEM = f"Figure_8/figure8_orientation_map_{QUANTITY}"

LABELS = {
    "area_abs": r"$\int_0^{1\,\mathrm{mT}}|\Delta\Phi_\alpha|\,\mathrm{d}B_0$ (mT)",
    "max_abs": r"$\max_{B_0}|\Delta\Phi_\alpha|$",
}

d = np.load(INPUT_FILE)
thetas = np.degrees(d["thetas"])
phis = np.degrees(d["phis"])
data = d[QUANTITY]                      # shape (N_theta, N_phi, 3)

dth = thetas[1] - thetas[0]
dph = phis[1] - phis[0]
extent = [phis[0] - dph / 2, phis[-1] + dph / 2, thetas[-1] + dth / 2, thetas[0] - dth / 2]
vmax = data.max()

# Principal axes and the off-axis orientation used in the Results
axis_marks = [
    ((0, 0), "z"), ((0, 180), "z"),
    ((0, 90), "x"), ((180, 90), "x"),
    ((90, 90), "y"), ((270, 90), "y"),
]
offaxis = (45, 45)

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True, constrained_layout=True)

for k, (ax, name) in enumerate(zip(axes, ("x", "y", "z"))):
    im = ax.imshow(data[:, :, k], extent=extent, aspect="auto", cmap="Blues",
                   vmin=0, vmax=vmax, interpolation="nearest")
    for (ph, th), lab in axis_marks:
        ax.plot(ph, th, "o", ms=4, mfc="white", mec="black", mew=0.6, clip_on=False)
        ax.annotate(lab, (ph, th), xytext=(3, -3), textcoords="offset points", fontsize=7)
    ax.plot(*offaxis, "^", ms=4.5, mfc="white", mec="black", mew=0.6)
    ax.set_title(rf"$P_{name}=1$", fontsize=8)
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.tick_params(width=0.6, length=2.5)

axes[0].set_ylabel(r"$\theta$ (deg)")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
cb.set_label(LABELS[QUANTITY])
cb.outline.set_linewidth(0.6)

fig.savefig(OUTPUT_STEM + ".pdf")
fig.savefig(OUTPUT_STEM + ".png", dpi=300)
print(f"Saved {OUTPUT_STEM}.pdf/.png")

# Values at the marked orientations, for the text
it = lambda th: np.argmin(np.abs(thetas - th))
ip = lambda ph: np.argmin(np.abs(phis - ph))
for (ph, th), lab in [((0, 0), "B||z"), ((0, 90), "B||x"), ((90, 90), "B||y"), (offaxis, "theta=phi=45")]:
    print(f"{lab:14s} {QUANTITY} (x, y, z):", np.round(data[it(th), ip(ph)], 5))