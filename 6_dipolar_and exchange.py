"""
Scenario 6: Inter-radical Couplings
Evaluates Isotropic and Anisotropic tensors with isolated and combined 
Exchange (J) and Dipolar (D) couplings.
Outputs two 4x4 subplot grids.
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
# Hyperfine Tensors
Axx, Ayy, Azz = 0.2, 0.03, 1.0
A_aniso = np.array([
    [ Axx, 0.0, 0.0],
    [ 0.0, Ayy, 0.0],
    [ 0.0, 0.0, Azz]
])

a_iso = (Axx + Ayy + Azz) / 3.0
A_iso = np.array([
    [ a_iso, 0.0, 0.0],
    [ 0.0, a_iso, 0.0],
    [ 0.0, 0.0, a_iso]
])

# Coupling Parameters
J_VAL = 1.0  # mT
D_VAL = 1.0  # mT

# Standard traceless axially symmetric Dipolar tensor aligned with Z
D_TENSOR_FIXED = np.array([
    [-D_VAL / 2, 0.0, 0.0],
    [0.0, -D_VAL / 2, 0.0],
    [0.0, 0.0, D_VAL]
])

# Coupling Conditions: (Name, J_ex, D_tensor)
CONDITIONS = [
    ('Base (J=0, D=0)', 0.0, None),
    ('Exchange (J=1)', J_VAL, None),
    ('Dipolar (D=1)', 0.0, D_TENSOR_FIXED),
    ('Coupled (J=1, D=1)', J_VAL, D_TENSOR_FIXED)
]

D_SPINS = [0.5]
A_SPINS = []

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
    t_name, hf_tensor, cond_name, j_ex, d_tensor, theta, phi = args
    
    builder = RPMBuilder(d_spins=D_SPINS, a_spins=A_SPINS,
                         a_tensor_d=[hf_tensor], a_tensor_a=[],
                         d_tensor=d_tensor, j_ex=j_ex,
                         k_s=K_S, k_t=K_T)

    y_low, y_x, y_y, y_z = [], [], [], []
    
    for B0 in B0_ARRAY:
        y_low.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis='z', theta=theta, phi=phi)[0])
        y_x.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=theta, phi=phi)[0])
        y_y.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=theta, phi=phi)[0])
        y_z.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=theta, phi=phi)[0])

    return t_name, cond_name, theta, phi, y_low, y_x, y_y, y_z

if __name__ == '__main__':
    start_time = time.perf_counter()
    angles = get_discrete_angles()
    
    # Build 4D grid of arguments: 2 Tensors * 4 Conditions * 4 Angles = 32 sweeps
    all_args = []
    for t_name, hf_tensor in [('Isotropic', A_iso), ('Anisotropic', A_aniso)]:
        for c_name, j_ex, d_tensor in CONDITIONS:
            for t, p in angles:
                all_args.append((t_name, hf_tensor, c_name, j_ex, d_tensor, t, p))

    print(f"Executing {len(all_args)} sweeps across J and D conditions...")

    cores = min(16, mp.cpu_count())
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_curve, all_args)

    # Organize data: dict[Tensor][Angle][Condition]
    data_dict = {
        'metadata': {'B0_ARRAY': B0_ARRAY},
        'Isotropic': {angle: {} for angle in angles},
        'Anisotropic': {angle: {} for angle in angles}
    }

    for res in results:
        t_name, c_name, theta, phi, y_low, y_x, y_y, y_z = res
        data_dict[t_name][(theta, phi)][c_name] = {
            'y_low': y_low, 'y_x': y_x, 'y_y': y_y, 'y_z': y_z
        }

    fname = 'scenario6_data.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

    # -------------------------------------------------------------------------
    # Plotting Helper Function
    # -------------------------------------------------------------------------
    def render_4x4_grid(tensor_name, data_subset, out_filename):
        fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)
        lw = 1.5

        for row, angle in enumerate(angles):
            theta, phi = angle
            angle_str = f"θ={theta/np.pi:.2f}π, φ={phi/np.pi:.2f}π"
            
            for col, (c_name, _, _) in enumerate(CONDITIONS):
                ax = axes[row, col]
                plot_data = data_subset[angle][c_name]
                
                # Only put legend on the very first subplot
                do_label = (row == 0 and col == 0)
                
                ax.plot(B0_ARRAY, plot_data['y_low'], color='black', lw=lw, ls='--', label='Unpol' if do_label else None)
                ax.plot(B0_ARRAY, plot_data['y_x'],   color='C0', lw=lw, label='Pol X' if do_label else None)
                ax.plot(B0_ARRAY, plot_data['y_y'],   color='C1', lw=lw, label='Pol Y' if do_label else None)
                ax.plot(B0_ARRAY, plot_data['y_z'],   color='C2', lw=lw, label='Pol Z' if do_label else None)

                ax.set_xscale('log')
                
                # Set Titles on Top Row
                if row == 0:
                    ax.set_title(c_name, fontsize=12, fontweight='bold')
                
                # Set Y-labels on Left Column
                if col == 0:
                    ax.set_ylabel(f"{angle_str}\n\nSinglet Yield ($\Phi_S$)")
                
                # Set X-labels on Bottom Row
                if row == 3:
                    ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
                    
                if do_label:
                    ax.legend(loc='lower right', frameon=False, fontsize='small')

        plt.suptitle(f'{tensor_name} Hyperfine: Effects of Exchange and Dipolar Coupling', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for suptitle
        plt.savefig(out_filename, dpi=200, bbox_inches='tight')
        plt.close()

    # Render plots
    render_4x4_grid('Isotropic', data_dict['Isotropic'], 'scenario6_isotropic.png')
    render_4x4_grid('Anisotropic', data_dict['Anisotropic'], 'scenario6_anisotropic.png')

    print(f"Scenario 6 complete in {time.perf_counter()-start_time:.2f}s.")
    print("Saved: scenario6_isotropic.png, scenario6_anisotropic.png")