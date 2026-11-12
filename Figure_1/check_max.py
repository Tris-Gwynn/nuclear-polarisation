import numpy as np
from scipy.signal import find_peaks

FILES = {
    "x": "Figure_1/screening_data_x.npz",
    "y": "Figure_1/screening_data_y.npz",
    "z": "Figure_1/screening_data_z.npz",
}
PROMINENCE_FRAC = 0.05 
SAVE_CSV = True

KINDS = {1.0: "peak", -1.0: "trough"}   


def refine(B0, y, i):
    """Parabolic interpolation in index space, mapped back to B0.
    Works for linear or log-spaced B0 grids."""
    if i <= 0 or i >= len(y) - 1:
        return B0[i], y[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return B0[i], y[i]
    p = 0.5 * (y0 - y2) / denom
    B0_ref = np.interp(i + p, np.arange(len(B0)), B0)
    y_ref = y1 - 0.25 * (y0 - y2) * p
    return B0_ref, y_ref


def extrema_1d(B0, m, sign):
    """Interior peaks (sign = +1) or troughs (sign = -1) of m along B0."""
    y = sign * np.asarray(m, dtype=float)
    rng = np.nanmax(y) - np.nanmin(y)
    prom = PROMINENCE_FRAC * rng if rng > 0 else None
    idx, props = find_peaks(y, prominence=prom)

    rows = []
    for k, i in enumerate(idx):
        B0_ref, y_ref = refine(B0, y, i)
        rows.append((sign, i, B0[i], m[i], props["prominences"][k], B0_ref, sign * y_ref))
    return rows


def global_extremum(B0, m, sign):
    """find_peaks ignores boundary points, so report the global max/min separately."""
    y = sign * np.asarray(m, dtype=float)
    g = int(np.nanargmax(y))
    return g, B0[g], m[g], g == 0 or g == len(y) - 1


def process(label, path):
    data = np.load(path)
    print(f"\n=== {label}: {path} ===")
    for key in data.files:
        print(f"  {key}: shape {data[key].shape}")

    B0 = np.asarray(data["B0_values"]).ravel()
    m = np.asarray(data["m_values"])

    matches = [ax for ax, n in enumerate(m.shape) if n == len(B0)]
    if not matches:
        raise ValueError(f"No axis of m_values {m.shape} matches len(B0_values) = {len(B0)}")
    if len(matches) > 1:
        print(f"  Warning: several axes match len(B0); using axis {matches[-1]}")
    m = np.moveaxis(m, matches[-1], -1)

    out = []
    for lead in np.ndindex(m.shape[:-1]):
        curve = m[lead]
        rows = extrema_1d(B0, curve, 1.0) + extrema_1d(B0, curve, -1.0)
        rows.sort(key=lambda r: r[1])   # order by B0 index
        n_pk = sum(r[0] > 0 for r in rows)
        tag = f"slice {lead}" if lead else "curve"
        print(f"  {tag}: {n_pk} peak(s), {len(rows) - n_pk} trough(s)")

        for sign, i, b, v, pr, b_ref, v_ref in rows:
            print(f"    {KINDS[sign]:6s}  idx {i:5d}  B0 = {b:.6g}  m = {v:.6g}  prom = {pr:.3g}  "
                  f"(refined B0 = {b_ref:.6g}, m = {v_ref:.6g})")
            out.append((*lead, sign, i, b, v, pr, b_ref, v_ref))

        for sign, name in ((1.0, "global max"), (-1.0, "global min")):
            g, b, v, edge = global_extremum(B0, curve, sign)
            note = "  <- at boundary of B0 range" if edge else ""
            print(f"    {name}  idx {g}  B0 = {b:.6g}  m = {v:.6g}{note}")

    if SAVE_CSV and out:
        lead_cols = [f"dim{d}" for d in range(m.ndim - 1)]
        header = ",".join(lead_cols + ["type(+1=peak,-1=trough)", "idx", "B0", "m",
                                       "prominence", "B0_refined", "m_refined"])
        np.savetxt(f"extrema_{label}.csv", np.array(out, dtype=float),
                   delimiter=",", header=header, comments="")
    return out


if __name__ == "__main__":
    results = {label: process(label, path) for label, path in FILES.items()}
    