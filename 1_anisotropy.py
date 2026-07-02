"""
Scenario 1: Isotropic vs. Anisotropic Hyperfine Comparison
Evaluates 4 discrete orientations for both tensors and plots them together.
"""
import time
import pickle
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RPM_Experiment_Builder import RPMBuilder

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
Axx, Ayy, Azz = 0.01, 0.1, 1.0
A_aniso = np.array([
    [ Axx, 0.0, 0.0],
    [ 0.0, Ayy, 0.0],
    [ 0.0, 0.0, Azz]
])

# Calculate trace-equivalent isotropic tensor
a_iso = (Axx + Ayy + Azz) / 3.0
A_iso = np.array([
    [ a_iso, 0.0, 0.0],
    [ 0.0, a_iso, 0.0],
    [ 0.0, 0.0, a_iso]
])

D_SPINS = [0.5]
A_SPINS = []
J_EX = 0.0
D_TENSOR = None

T_MAX = 5.0
P_LOW = 0.0
P_HIGH = 1.0
K_S, K_T = 1.0, 1.0

B0_MIN, B0_MAX, B0_POINTS = 1e-3, 10.0, 100
B0_ARRAY = np.logspace(np.log10(B0_MIN), np.log10(B0_MAX), B0_POINTS)

def get_discrete_angles():
    return [
        (0.0, 0.0),             # Z-axis
        (np.pi/2, 0.0),         # X-axis
        (np.pi/2, np.pi/2),     # Y-axis
        (np.pi/4, np.pi/4)      # 45-degree off-axis
    ]

def compute_mary_curve(args):
    tensor, theta, phi = args
    builder = RPMBuilder(d_spins=D_SPINS, a_spins=A_SPINS,
                         a_tensor_d=[tensor], a_tensor_a=[],
                         d_tensor=D_TENSOR, j_ex=J_EX,
                         k_s=K_S, k_t=K_T)

    y_low, y_x, y_y, y_z = [], [], [], []
    
    for B0 in B0_ARRAY:
        y_low.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis='z', theta=theta, phi=phi)[0])
        y_x.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=theta, phi=phi)[0])
        y_y.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=theta, phi=phi)[0])
        y_z.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=theta, phi=phi)[0])

    return y_low, y_x, y_y, y_z

if __name__ == '__main__':
    start_time = time.perf_counter()
    angles = get_discrete_angles()
    
    # Build argument lists for both tensor types
    aniso_args = [(A_aniso, t, p) for t, p in angles]
    iso_args = [(A_iso, t, p) for t, p in angles]
    all_args = aniso_args + iso_args

    print(f"Sweeping {B0_POINTS} points for Anisotropic and Isotropic tensors...")

    cores = 20
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_curve, all_args)

    # Split results back into aniso and iso
    results_aniso = results[:len(angles)]
    results_iso = results[len(angles):]

    # Pack data
    data_dict = {
        'metadata': {'B0_ARRAY': B0_ARRAY, 'P_LOW': P_LOW, 'P_HIGH': P_HIGH, 'a_iso_val': a_iso},
        'anisotropic': [{'theta': t, 'phi': p, 'data': res} for (t, p), res in zip(angles, results_aniso)],
        'isotropic': [{'theta': t, 'phi': p, 'data': res} for (t, p), res in zip(angles, results_iso)]
    }

    fname = 'scenario1_data.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 7))
    lw = 1.5

    # Plot Anisotropic (Solid lines)
    for i, item in enumerate(data_dict['anisotropic']):
        y_low, y_x, y_y, y_z = item['data']
        lbl_prefix = "Aniso" if i == 0 else ""
        
        ax1.plot(B0_ARRAY, y_low, color='black', lw=lw, ls='-', label=f'{lbl_prefix} Unpol' if i==0 else None)
        ax1.plot(B0_ARRAY, y_x,   color='C0', lw=lw, ls='-', label=f'{lbl_prefix} Pol X' if i==0 else None)
        ax1.plot(B0_ARRAY, y_y,   color='C1', lw=lw, ls='-', label=f'{lbl_prefix} Pol Y' if i==0 else None)
        ax1.plot(B0_ARRAY, y_z,   color='C2', lw=lw, ls='-', label=f'{lbl_prefix} Pol Z' if i==0 else None)

    # Plot Isotropic (Dashed lines)
    for i, item in enumerate(data_dict['isotropic']):
        y_low, y_x, y_y, y_z = item['data']
        lbl_prefix = "Iso" if i == 0 else ""
        
        ax1.plot(B0_ARRAY, y_low, color='black', lw=lw, ls='--', alpha=0.7, label=f'{lbl_prefix} Unpol' if i==0 else None)
        ax1.plot(B0_ARRAY, y_x,   color='C0', lw=lw, ls='--', alpha=0.7, label=f'{lbl_prefix} Pol X' if i==0 else None)
        ax1.plot(B0_ARRAY, y_y,   color='C1', lw=lw, ls='--', alpha=0.7, label=f'{lbl_prefix} Pol Y' if i==0 else None)
        ax1.plot(B0_ARRAY, y_z,   color='C2', lw=lw, ls='--', alpha=0.7, label=f'{lbl_prefix} Pol Z' if i==0 else None)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax1.set_xlim(B0_MIN, B0_MAX)
    ax1.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')
    ax1.set_ylim(0.0, 1.0)
    
    ax1.legend(loc='lower right', frameon=False, ncol=2)
    plt.title(f'MARY Comparison: Anisotropic vs Isotropic ($a_{{iso}} = {a_iso:.3f}$ mT)')
    plt.tight_layout()

    out_fname = 'scenario1_mary.png'
    plt.savefig(out_fname, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Scenario 1 complete in {time.perf_counter()-start_time:.2f}s.")
    print(f"Data saved to {fname}, Plot saved to {out_fname}")