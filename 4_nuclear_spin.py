"""
Scenario 5: Spin Quantum Number (Spin-1/2 vs Spin-1)
Evaluates an isotropic hyperfine tensor across 4 orientations, comparing
the effect of the nuclear spin quantum number.
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
Axx_base, Ayy_base, Azz_base = 0.2, 0.03, 1.0
a_iso = (Axx_base + Ayy_base + Azz_base) / 3.0
A_iso = np.array([
    [ a_iso, 0.0, 0.0],
    [ 0.0, a_iso, 0.0],
    [ 0.0, 0.0, a_iso]
])

SPIN_VALUES = [0.5, 1.0]

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
    spin_val, theta, phi = args
    
    builder = RPMBuilder(d_spins=[spin_val], a_spins=[],
                         a_tensor_d=[A_iso], a_tensor_a=[],
                         d_tensor=D_TENSOR, j_ex=J_EX,
                         k_s=K_S, k_t=K_T)

    y_low, y_x, y_y, y_z = [], [], [], []
    
    for B0 in B0_ARRAY:
        y_low.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis='z', theta=theta, phi=phi)[0])
        y_x.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=theta, phi=phi)[0])
        y_y.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=theta, phi=phi)[0])
        y_z.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=theta, phi=phi)[0])

    return spin_val, theta, phi, y_low, y_x, y_y, y_z

if __name__ == '__main__':
    start_time = time.perf_counter()
    angles = get_discrete_angles()
    
    # Build argument list: evaluate both spin values at all 4 angles
    all_args = [(s, t, p) for s in SPIN_VALUES for t, p in angles]

    print(f"Executing {len(all_args)} sweeps for Spin-1/2 vs Spin-1 comparison...")

    cores = min(8, mp.cpu_count())
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_curve, all_args)

    # Organize data by angle, then by spin value
    data_dict = {
        'metadata': {'B0_ARRAY': B0_ARRAY, 'P_LOW': P_LOW, 'P_HIGH': P_HIGH, 'a_iso': a_iso},
        'results': {angle: {} for angle in angles}
    }

    for res in results:
        spin_val, theta, phi, y_low, y_x, y_y, y_z = res
        data_dict['results'][(theta, phi)][spin_val] = {
            'y_low': y_low, 'y_x': y_x, 'y_y': y_y, 'y_z': y_z
        }

    fname = 'scenario5_data.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    lw = 1.5

    for ax, angle in zip(axes, angles):
        theta, phi = angle
        angle_data = data_dict['results'][angle]
        
        # Plot Spin 1/2 (Solid lines)
        s05 = angle_data[0.5]
        ax.plot(B0_ARRAY, s05['y_low'], color='black', lw=lw, ls='-', label=r'Unpol ($I=1/2$)')
        ax.plot(B0_ARRAY, s05['y_x'],   color='C0', lw=lw, ls='-', label=r'Pol X ($I=1/2$)')
        ax.plot(B0_ARRAY, s05['y_y'],   color='C1', lw=lw, ls='-', label=r'Pol Y ($I=1/2$)')
        ax.plot(B0_ARRAY, s05['y_z'],   color='C2', lw=lw, ls='-', label=r'Pol Z ($I=1/2$)')

        # Plot Spin 1 (Dashed lines, slightly thicker for visibility)
        s10 = angle_data[1.0]
        ax.plot(B0_ARRAY, s10['y_low'], color='black', lw=lw+0.5, ls='--', alpha=0.8, label=r'Unpol ($I=1$)')
        ax.plot(B0_ARRAY, s10['y_x'],   color='C0', lw=lw+0.5, ls='--', alpha=0.8, label=r'Pol X ($I=1$)')
        ax.plot(B0_ARRAY, s10['y_y'],   color='C1', lw=lw+0.5, ls='--', alpha=0.8, label=r'Pol Y ($I=1$)')
        ax.plot(B0_ARRAY, s10['y_z'],   color='C2', lw=lw+0.5, ls='--', alpha=0.8, label=r'Pol Z ($I=1$)')

        ax.set_xscale('log')
        ax.set_title(f"θ={theta/np.pi:.2f}π, φ={phi/np.pi:.2f}π")
        
        # Only add legend to the first plot to avoid clutter
        if ax == axes[0]:
            ax.legend(loc='lower right', frameon=False, fontsize='small', ncol=2)
            
        if ax in [axes[2], axes[3]]:
            ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
        if ax in [axes[0], axes[2]]:
            ax.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')

    plt.suptitle(f'Isotropic MARY ($a_{{iso}}={a_iso:.3f}$ mT): Spin-1/2 vs Spin-1', fontsize=14)
    plt.tight_layout()
    
    out_fname = 'scenario5_spin_quantum.png'
    plt.savefig(out_fname, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Scenario 5 complete in {time.perf_counter()-start_time:.2f}s.")
    print(f"Saved to {out_fname}")