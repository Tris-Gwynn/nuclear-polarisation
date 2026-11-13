
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "Figure_8/figure8_orientation_map.npz"
QUANTITY = "area_abs"
OUTPUT_STEM = f"Figure_8/figure8_reversal_{QUANTITY}"

d = np.load(INPUT_FILE)
thetas = np.degrees(d["thetas"].astype(float))
phis = np.degrees(d["phis"].astype(float))
A = d[QUANTITY].astype(float)                      # (N_theta, N_phi, 3)
Nt, Np = len(thetas), len(phis)

# -n has theta -> 180 - theta and phi -> phi + 180
it_rev = Nt - 1 - np.arange(Nt)
ip_rev = (np.arange(Np) + Np // 2) % Np
A_rev = A[it_rev][:, ip_rev]
D = A - A_rev

upper = thetas <= 90.0
Du, tu = D[upper], thetas[upper]
vmax = np.abs(Du).max()

print("max |D| per polarisation (x, y, z):", np.round(np.abs(Du).reshape(-1, 3).max(0), 5))
print("max |D| / max A per polarisation:   ", np.round(np.abs(Du).reshape(-1, 3).max(0) / A.reshape(-1, 3).max(0), 3))
on_planes = np.concatenate([Du[:, np.isin(np.round(phis, 6), [0, 90, 180, 270])].ravel(), D[np.isclose(thetas, 90)].ravel()])
print("max |D| on the principal planes:    ", np.abs(on_planes).max())

dth, dph = tu[1] - tu[0], phis[1] - phis[0]
extent = [phis[0] - dph / 2, phis[-1] + dph / 2, tu[-1] + dth / 2, tu[0] - dth / 2]

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True, constrained_layout=True)
for k, (ax, name) in enumerate(zip(axes, ("x", "y", "z"))):
    im = ax.imshow(Du[:, :, k], extent=extent, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    for (ph, th), lab in (((0, 90), "x"), ((180, 90), "x"), ((90, 90), "y"), ((270, 90), "y"), ((0, 0), "z")):
        ax.plot(ph, th, "o", ms=4, mfc="white", mec="black", mew=0.6, clip_on=False)
        ax.annotate(lab, (ph, th), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.plot(45, 45, "^", ms=4.5, mfc="white", mec="black", mew=0.6)
    ax.set_title(rf"$P_{name}=1$", fontsize=8)
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 30, 60, 90])
    ax.tick_params(width=0.6, length=2.5)
axes[0].set_ylabel(r"$\theta$ (deg)")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
cb.set_label(r"$A_\alpha(\hat{\mathbf{n}})-A_\alpha(-\hat{\mathbf{n}})$ (mT)")
cb.outline.set_linewidth(0.6)

fig.savefig(OUTPUT_STEM + ".pdf")
fig.savefig(OUTPUT_STEM + ".png", dpi=300)
print(f"Saved {OUTPUT_STEM}.pdf/.png")