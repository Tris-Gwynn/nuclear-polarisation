"""
Plot the orientation map on the unit sphere of field directions.

Each sphere shows the field-integrated |Delta Phi_alpha| for one polarisation
direction, as a function of the field direction in the hyperfine principal-axis
frame. Principal axes are drawn through the sphere, and the off-axis direction
theta = phi = pi/4 used in the Results is marked. All spheres share one colour
scale.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

INPUT_FILE = "Figure_8/figure8_orientation_map.npz"
QUANTITY = "area_abs"
OUTPUT_STEM = f"Figure_8/figure8_sphere_{QUANTITY}"
STRIDE = 2                  # plot every STRIDE-th grid point (0.5 deg grid -> 1 deg mesh)
VIEW = dict(elev=25, azim=35)

LABELS = {
    "area_abs": r"$\int_0^{1\,\mathrm{mT}}|\Delta\Phi_\alpha|\,\mathrm{d}B_0$ (mT)",
    "max_abs": r"$\max_{B_0}|\Delta\Phi_\alpha|$",
}

d = np.load(INPUT_FILE)
th = d["thetas"].astype(float)[::STRIDE]
ph = d["phis"].astype(float)[::STRIDE]
data = d[QUANTITY].astype(float)[::STRIDE, ::STRIDE]

# Close the sphere in phi by repeating the phi = 0 column at phi = 2 pi
ph = np.append(ph, 2 * np.pi)
data = np.concatenate([data, data[:, :1]], axis=1)

TH, PH = np.meshgrid(th, ph, indexing="ij")
X, Y, Z = np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)

norm = colors.Normalize(vmin=0, vmax=data.max())
cmap = cm.Blues

plt.rcParams.update({"font.size": 8})
fig = plt.figure(figsize=(7.0, 2.5))
fig.subplots_adjust(left=0.0, right=0.88, bottom=0.1, top=1.0, wspace=0.0)

for k, name in enumerate(("x", "y", "z")):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(data[:, :, k])), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    for v, lab in ((np.array([1, 0, 0]), "x"), (np.array([0, 1, 0]), "y"), (np.array([0, 0, 1]), "z")):
        ax.plot(*np.c_[1.0 * v, 1.22 * v], color="black", lw=0.8, zorder=5)   # outward stub on the visible side
        ax.text(*(1.32 * v), lab, fontsize=8, ha="center", va="center", zorder=6)
    n45 = np.array([np.sin(np.pi / 4) * np.cos(np.pi / 4), np.sin(np.pi / 4) * np.sin(np.pi / 4), np.cos(np.pi / 4)])
    ax.scatter(*(1.01 * n45), marker="^", s=18, facecolor="white", edgecolor="black", linewidth=0.6, depthshade=False, zorder=7)
    ax.view_init(**VIEW)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.text2D(0.5, 0.0, rf"$P_{name}=1$", transform=ax.transAxes, ha="center", va="top", fontsize=8)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cb = fig.colorbar(sm, cax=cax)
cb.set_label(LABELS[QUANTITY])
cb.outline.set_linewidth(0.6)

fig.savefig(OUTPUT_STEM + ".pdf", bbox_inches="tight")
fig.savefig(OUTPUT_STEM + ".png", dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_STEM}.pdf/.png")