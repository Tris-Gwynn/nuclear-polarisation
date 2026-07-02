"""
Scenario 3: Hyperfine Tensor Rotation vs Zeeman Orientation
Evaluates 4 B0 orientations against 4 Hyperfine tensor orientations.
Outputs 4 separate plots (one for each B0 angle).
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
A_aniso_base = np.array([
    [ Axx_base, 0.0, 0.0],
    [ 0.0, Ayy_base, 0.0],
    [ 0.0, 0.0, Azz_base]
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

def get_rotation_matrix(theta, phi):
    """Generates a 3D rotation matrix R_z(phi) * R_y(theta)."""
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_z = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y

def compute_mary_curve(args):
    b_theta, b_phi, hf_theta, hf_phi = args
    
    # Rotate the Hyperfine Tensor
    R = get_rotation_matrix(hf_theta, hf_phi)
    A_rot = R @ A_aniso_base @ R.T

    builder = RPMBuilder(d_spins=D_SPINS, a_spins=A_SPINS,
                         a_tensor_d=[A_rot], a_tensor_a=[],
                         d_tensor=D_TENSOR, j_ex=J_EX,
                         k_s=K_S, k_t=K_T)

    y_low, y_x, y_y, y_z = [], [], [], []
    
    for B0 in B0_ARRAY:
        y_low.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis='z', theta=b_theta, phi=b_phi)[0])
        y_x.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=b_theta, phi=b_phi)[0])
        y_y.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=b_theta, phi=b_phi)[0])
        y_z.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=b_theta, phi=b_phi)[0])

    return b_theta, b_phi, hf_theta, hf_phi, y_low, y_x, y_y, y_z

if __name__ == '__main__':
    start_time = time.perf_counter()
    angles = get_discrete_angles()
    
    # Create the 4x4 parameter grid
    all_args = [(b_t, b_p, hf_t, hf_p) for b_t, b_p in angles for hf_t, hf_p in angles]

    print(f"Executing {len(all_args)} sweeps for B0/HF angle combinations...")

    cores = min(16, mp.cpu_count())
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_curve, all_args)

    # Organize data: dict keyed by B0 angle, containing a list of HF results
    data_dict = {
        'metadata': {'B0_ARRAY': B0_ARRAY, 'P_LOW': P_LOW, 'P_HIGH': P_HIGH},
        'results': {b_angle: [] for b_angle in angles}
    }

    for res in results:
        b_t, b_p, hf_t, hf_p, y_low, y_x, y_y, y_z = res
        data_dict['results'][(b_t, b_p)].append({
            'hf_theta': hf_t, 'hf_phi': hf_p,
            'y_low': y_low, 'y_x': y_x, 'y_y': y_y, 'y_z': y_z
        })

    fname = 'scenario3_data.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

    # -------------------------------------------------------------------------
    # Plotting Helper Function
    # -------------------------------------------------------------------------
    def render_zeeman_plot(b_angle, hf_data_list, out_filename):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        pol_states = ['y_low', 'y_x', 'y_y', 'y_z']
        pol_titles = ['Unpolarized', 'Polarized X', 'Polarized Y', 'Polarized Z']
        
        for ax, state, title in zip(axes, pol_states, pol_titles):
            for i, hf_item in enumerate(hf_data_list):
                hf_t, hf_p = hf_item['hf_theta'], hf_item['hf_phi']
                y_data = hf_item[state]
                
                lbl = f"HF θ={hf_t/np.pi:.2f}π, φ={hf_p/np.pi:.2f}π"
                ls = '-' if i == 0 else '--' if i == 1 else '-.' if i == 2 else ':'
                
                ax.plot(B0_ARRAY, y_data, lw=1.5, ls=ls, label=lbl)
                
            ax.set_xscale('log')
            ax.set_title(title)
            ax.legend(loc='lower right', frameon=False, fontsize='small')
            
            if ax in [axes[2], axes[3]]:
                ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
            if ax in [axes[0], axes[2]]:
                ax.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')
                
        b_t, b_p = b_angle
        plt.suptitle(f'Zeeman Orientation: θ={b_t/np.pi:.2f}π, φ={b_p/np.pi:.2f}π\nMARY Curves vs Hyperfine Rotation', fontsize=14)
        plt.tight_layout()
        plt.savefig(out_filename, dpi=200, bbox_inches='tight')
        plt.close()

    # Render a separate image for each B0 orientation
    for i, b_angle in enumerate(angles):
        out_name = f'scenario3_mary_B0_{i}.png'
        render_zeeman_plot(b_angle, data_dict['results'][b_angle], out_name)

    print(f"Scenario 3 complete in {time.perf_counter()-start_time:.2f}s.")
    print(f"Saved {len(angles)} plots (e.g., scenario3_mary_B0_0.png)")