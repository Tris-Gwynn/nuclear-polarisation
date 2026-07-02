"""
Loads the MARY dictionary file and renders the cloud plot.
"""
import pickle
import matplotlib

#from generate_mary import B0_ARRAY
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_data(data_file):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    meta = data_dict['metadata']
    B0_ARRAY = meta['B0_ARRAY']
    P_LOW = meta['P_LOW']
    P_HIGH = meta['P_HIGH']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    lw = 1.0
    alpha_cloud = 1.0

    # Iterate through the stored dictionary list
    for i, orientation in enumerate(data_dict['orientations']):
        y_low = orientation['y_low']
        y_x = orientation['y_x']
        y_y = orientation['y_y']
        y_z = orientation['y_z']

        lbl_low = rf'Unpolarized ($P={P_LOW:.0f}$)' if i == 0 else None
        lbl_x   = rf'Fully Polarized ($P_x={P_HIGH:.0f}$)' if i == 0 else None
        lbl_y   = rf'Fully Polarized ($P_y={P_HIGH:.0f}$)' if i == 0 else None
        lbl_z   = rf'Fully Polarized ($P_z={P_HIGH:.0f}$)' if i == 0 else None

        ax1.plot(B0_ARRAY, y_low, color='C3', lw=lw, alpha=alpha_cloud, label=lbl_low)
        ax1.plot(B0_ARRAY, y_x,   color='C0', lw=lw, alpha=alpha_cloud, label=lbl_x)
        ax1.plot(B0_ARRAY, y_y,   color='C1', lw=lw, alpha=alpha_cloud, label=lbl_y)
        ax1.plot(B0_ARRAY, y_z,   color='C2', lw=lw, alpha=alpha_cloud, label=lbl_z)
        #ax1.plot(B0_ARRAY, y_x-y_low, color='C0', lw=lw, alpha=alpha_cloud, label=lbl_x)
        #ax1.plot(B0_ARRAY, y_y-y_low, color='C1', lw=lw, alpha=alpha_cloud, label=lbl_y)
        #ax1.plot(B0_ARRAY, y_z-y_low, color='C3', lw=lw, alpha=alpha_cloud, label=lbl_z)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
    ax1.set_xlim(B0_ARRAY[0], B0_ARRAY[-1])
    ax1.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')
    ax1.set_ylim(0.0, 1.0)
    
    leg = ax1.legend(loc='lower right', frameon=False)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0) #type: ignore
        lh.set_linewidth(2.0) #type: ignore

    plt.title('Magnetically Altered Reaction Yield (MARY) - Orientation Cloud')
    plt.tight_layout()

    out_fname = 'mary_sweep_cloud.png'
    plt.savefig(out_fname, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Plot rendered and saved to {out_fname}")

if __name__ == '__main__':
    plot_data('mary_data.pkl')