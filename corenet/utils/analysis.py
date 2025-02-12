import numpy as np
import matplotlib.pyplot as plt


def compute_efficiencies(
    weak_values,
    gut_test_constraints,
    gut_test_output_constraints,
    bins=10,
):
    hist, x_edges, y_edges = np.histogram2d(
        weak_values[:, 0],
        weak_values[:, 1],
        bins=[bins, bins]
    )
    x_bin_indices = np.digitize(weak_values[:, 0], x_edges) - 1
    y_bin_indices = np.digitize(weak_values[:, 1], y_edges) - 1
    bin_entries = np.array(list(zip(x_bin_indices, y_bin_indices)))

    """Create empty histogram scalars for validities"""
    gut_test_m_H = np.zeros(hist.shape)
    gut_test_dm = np.zeros(hist.shape)
    gut_test_double = np.zeros(hist.shape)
    gut_test_triple = np.zeros(hist.shape)
    gut_test_output_m_H = np.zeros(hist.shape)
    gut_test_output_dm = np.zeros(hist.shape)
    gut_test_output_double = np.zeros(hist.shape)
    gut_test_output_triple = np.zeros(hist.shape)
    m_H_efficiency = np.zeros(hist.shape)
    dm_efficiency = np.zeros(hist.shape)
    double_efficiency = np.zeros(hist.shape)
    triple_efficiency = np.zeros(hist.shape)

    for i, x_edge in enumerate(x_bin_indices):
        if i >= bins:
            continue
        for j, y_edge in enumerate(y_bin_indices):
            if j >= bins:
                continue
            bin_ents = (bin_entries[:, 0] == i) & (bin_entries[:, 1] == j)
            gut_test_bin_constraints = gut_test_constraints[bin_ents]
            gut_test_output_bin_constraints = gut_test_output_constraints[bin_ents]
            """Make valid masks"""
            gut_test_m_H_mask = (gut_test_bin_constraints[:, 0] == 1.0)
            gut_test_dm_mask = (gut_test_bin_constraints[:, 1] == 1.0)
            gut_test_double_mask = (gut_test_m_H_mask & gut_test_dm_mask)
            gut_test_triple_mask = (gut_test_double_mask) & (gut_test_bin_constraints[:, 2] == 1.0)

            gut_test_output_m_H_mask = (gut_test_output_bin_constraints[:, 0] == 1.0)
            gut_test_output_dm_mask = (gut_test_output_bin_constraints[:, 1] == 1.0)
            gut_test_output_double_mask = (gut_test_output_m_H_mask & gut_test_output_dm_mask)
            gut_test_output_triple_mask = (gut_test_output_double_mask) & (gut_test_output_bin_constraints[:, 2] == 1.0)

            gut_test_m_H[i][j] = sum(gut_test_m_H_mask)
            gut_test_dm[i][j] = sum(gut_test_dm_mask)
            gut_test_double[i][j] = sum(gut_test_double_mask)
            gut_test_triple[i][j] = sum(gut_test_triple_mask)

            gut_test_output_m_H[i][j] = sum(gut_test_output_m_H_mask)
            gut_test_output_dm[i][j] = sum(gut_test_output_dm_mask)
            gut_test_output_double[i][j] = sum(gut_test_output_double_mask)
            gut_test_output_triple[i][j] = sum(gut_test_output_triple_mask)

            if sum(~gut_test_m_H_mask):
                m_H_efficiency[i][j] = sum(~gut_test_m_H_mask & gut_test_output_m_H_mask) / sum(~gut_test_m_H_mask)
            else:
                m_H_efficiency[i][j] = 1.0
            if sum(~gut_test_dm_mask):
                dm_efficiency[i][j] = sum(~gut_test_dm_mask & gut_test_output_dm_mask) / sum(~gut_test_dm_mask)
            else:
                dm_efficiency[i][j] = 1.0
            if sum(~gut_test_double_mask):
                double_efficiency[i][j] = sum(~gut_test_double_mask & gut_test_output_double_mask) / sum(~gut_test_double_mask)
            else:
                double_efficiency[i][j] = 1.0
            if sum(~gut_test_triple_mask):
                triple_efficiency[i][j] = sum(~gut_test_triple_mask & gut_test_output_triple_mask) / sum(~gut_test_triple_mask)
            else:
                triple_efficiency[i][j] = 1.0
    return {
        'hist': hist,
        'x_edges': x_edges,
        'y_edges': y_edges,
        'x_centers': [.5*(x_edges[ii] + x_edges[ii+1]) for ii in range(len(x_edges)-1)],
        'y_centers': [.5*(y_edges[ii] + y_edges[ii+1]) for ii in range(len(y_edges)-1)],
        'bin_entries': bin_entries,
        'gut_test_m_H': gut_test_m_H,
        'gut_test_dm': gut_test_dm,
        'gut_test_double': gut_test_double,
        'gut_test_triple': gut_test_triple,
        'gut_test_output_m_H': gut_test_output_m_H,
        'gut_test_output_dm': gut_test_output_dm,
        'gut_test_output_double': gut_test_output_double,
        'gut_test_output_triple': gut_test_output_triple,
        'm_H_efficiency': m_H_efficiency,
        'dm_efficiency': dm_efficiency,
        'double_efficiency': double_efficiency,
        'triple_efficiency': triple_efficiency,
    }


def make_single_model_analysis_plots(
    weak_test,
    weak_output,
    gut_test_constraints,
    gut_test_output_constraints,
    efficiencies,
    save_directory
):
    """Make various distribution plots for m_H and omega_dm"""
    fig, axs = plt.subplots(figsize=(10, 6))
    non_zero = (gut_test_output_constraints[:, 0] == 1)
    axs.scatter(
        weak_test[:, 0],
        weak_output[:, 0],
        alpha=0.5,
        s=1
    )
    axs.set_xlabel(r'$m_H$ input')
    axs.set_ylabel(r'$m_H$ reconstructed')
    axs.set_title(r'$m_H$ Reconstructed Distribution')
    axs.set_xlim([122.09, 128.09])
    axs.set_ylim([122.09, 128.09])
    plt.savefig(f'{save_directory}/m_H_reconstruction.png')
    plt.close()

    fig, axs = plt.subplots(figsize=(10, 6))
    non_zero = (gut_test_output_constraints[:, 1] == 1)
    axs.scatter(
        weak_test[:, 1],
        weak_output[:, 1],
        alpha=0.5,
        s=1
    )
    axs.set_xlabel(r'$\Omega_{DM}$ input')
    axs.set_ylabel(r'$\Omega_{DM}$ reconstructed')
    axs.set_title(r'$\Omega_{DM}$ Reconstructed Distribution')
    axs.set_xlim([0.08, 0.14])
    axs.set_ylim([0.08, 0.14])
    plt.savefig(f'{save_directory}/dm_reconstruction.png')
    plt.close()

    """Plot error histograms for m_H and dm reconstruction"""
    fig, axs = plt.subplots(figsize=(10, 6))
    non_zero = (gut_test_output_constraints[:, 0] == 1)
    """Compute error and square error"""
    error = (
        weak_test[:, 0] - weak_output[:, 0]
    )/(weak_test[:, 0])
    square_error = error ** 2
    mean_square_error = np.mean(square_error)
    """Plot histogram of error"""
    axs.hist(
        error,
        bins=100,
        histtype='step',
        color='r'
    )
    axs.scatter(
        [], [],
        label=r'$\langle E^2_{m_H} \rangle$:'+f' {mean_square_error:.2e}',
        c='k',
        s=0.0
    )
    axs.set_yscale('log')
    axs.set_xlabel(r'Reconstruction error')
    axs.set_title(r'$m_H$ Reconstruction Error $(m_H - \mathrm{cor}(m_H))/m_H$')
    axs.legend()
    plt.savefig(f'{save_directory}/m_H_reconstruction_error.png')
    plt.close()

    fig, axs = plt.subplots(figsize=(10, 6))
    non_zero = (gut_test_output_constraints[:, 1] == 1)
    """Compute error and square error"""
    error = (
        weak_test[:, 1] - weak_output[:, 1]
    )/(weak_test[:, 1])
    square_error = error ** 2
    mean_square_error = np.mean(square_error)
    """Plot histogram of error"""
    axs.hist(
        error,
        bins=100,
        histtype='step',
        color='r'
    )
    axs.scatter(
        [], [],
        label=r'$\langle E^2_{\Omega_{DM}} \rangle$:'+f' {mean_square_error:.2e}',
        c='k',
        s=0.0
    )
    axs.set_yscale('log')
    axs.set_xlabel(r'Reconstruction error')
    axs.set_title(r'$\Omega_{DM}$ Reconstruction Error $(\Omega_{DM} - \mathrm{cor}(\Omega_{DM}))/\Omega_{DM}$')
    axs.legend()
    plt.savefig(f'{save_directory}/dm_reconstruction_error.png')
    plt.close()

    """Plot the error as a function of input value"""
    fig, axs = plt.subplots(figsize=(10, 6))
    non_zero = (gut_test_output_constraints[:, 0] == 1)
    x_values = weak_test[:, 0]
    error = (
        weak_test[:, 0] - weak_output[:, 0]
    )/(weak_test[:, 0])
    h = axs.hist2d(
        x_values,
        error,
        bins=(50, 50),
        cmap="Blues",
        density=True
    )
    plt.colorbar(h[3], ax=axs, label="Density")  # Add colorbar
    axs.set_xlabel(r'$m_H$ input')
    axs.set_ylabel(r'$m_H$ Reconstruction Error $(m_H - \mathrm{cor}(m_H))/m_H$')
    axs.set_title(r'$m_H$ Reconstruction Error vs. $m_H$')
    plt.savefig(f'{save_directory}/m_H_reconstruction_error_per_m_H_hist.png')
    plt.close()

    fig, axs = plt.subplots()
    non_zero = (gut_test_output_constraints[:, 0] == 1)
    x_values = weak_test[:, 0]
    error = (
        weak_test[:, 0] - weak_output[:, 0]
    )/(weak_test[:, 0])
    # Compute mean and standard deviation in bins
    bins = np.linspace(
        x_values.min(),
        x_values.max(),
        50
    )  # Adjust bin count as needed
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = np.array([
        error[(x_values >= bins[i]) & (x_values < bins[i+1])].mean()
        for i in range(len(bins) - 1)
    ])
    bin_stds = np.array([
        error[(x_values >= bins[i]) & (x_values < bins[i+1])].std()
        for i in range(len(bins) - 1)
    ])
    # Plot mean line
    axs.plot(bin_centers, bin_means, color="red", lw=2, label="Mean")
    # Plot shaded 1σ bands
    axs.fill_between(
        bin_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        color="red",
        alpha=0.3,
        label=r'$\pm 1\sigma$'
    )
    axs.set_xlabel(r'$m_H$ input')
    axs.set_ylabel(r'$m_H$ Reconstruction Error $(m_H - \mathrm{cor}(m_H))/m_H$')
    axs.set_title(r'$m_H$ Reconstruction Error vs. $m_H$')
    plt.savefig(f'{save_directory}/m_H_reconstruction_error_per_m_H_linear.png')
    plt.close()

    """Plot the error as a function of input value"""
    fig, axs = plt.subplots(figsize=(10, 6))
    non_zero = (gut_test_output_constraints[:, 1] == 1)
    x_values = weak_test[:, 1]
    error = (
        weak_test[:, 1] - weak_output[:, 1]
    )/(weak_test[:, 1])
    h = axs.hist2d(
        x_values,
        error,
        bins=(50, 50),
        cmap="Blues",
        density=True
    )
    plt.colorbar(h[3], ax=axs, label="Density")  # Add colorbar
    axs.set_xlabel(r'$\Omega_{DM}$ input')
    axs.set_ylabel(r'$\Omega_{DM}$ Reconstruction Error $(\Omega_{DM} - \mathrm{cor}(\Omega_{DM}))/\Omega_{DM}$')
    axs.set_title(r'$\Omega_{DM}$ Reconstruction Error vs. $\Omega_{DM}$')
    plt.savefig(f'{save_directory}/dm_reconstruction_error_per_dm_hist.png')
    plt.close()

    fig, axs = plt.subplots()
    non_zero = (gut_test_output_constraints[:, 1] == 1)
    x_values = weak_test[:, 1]
    error = (
        weak_test[:, 1] - weak_output[:, 1]
    )/(weak_test[:, 1])
    # Compute mean and standard deviation in bins
    bins = np.linspace(
        x_values.min(),
        x_values.max(),
        50
    )  # Adjust bin count as needed
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = np.array([
        error[(x_values >= bins[i]) & (x_values < bins[i+1])].mean()
        for i in range(len(bins) - 1)
    ])
    bin_stds = np.array([
        error[(x_values >= bins[i]) & (x_values < bins[i+1])].std()
        for i in range(len(bins) - 1)
    ])
    # Plot mean line
    axs.plot(bin_centers, bin_means, color="red", lw=2, label="Mean")
    # Plot shaded 1σ bands
    axs.fill_between(
        bin_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        color="red",
        alpha=0.3,
        label=r'$\pm 1\sigma$'
    )
    axs.set_xlabel(r'$\Omega_{DM}$ input')
    axs.set_ylabel(r'$\Omega_{DM}$ Reconstruction Error $(\Omega_{DM} - \mathrm{cor}(\Omega_{DM}))/\Omega_{DM}$')
    axs.set_title(r'$\Omega_{DM}$ Reconstruction Error vs. $\Omega_{DM}$')
    plt.savefig(f'{save_directory}/dm_reconstruction_error_per_dm_linear.png')
    plt.close()

    """Make Efficiency plots"""
    fig, axs = plt.subplots(figsize=(10, 6))
    m_H_efficiences = np.mean(efficiencies['m_H_efficiency'], axis=0)
    axs.plot(
        efficiencies['x_centers'],
        m_H_efficiences,
        linestyle='--'
    )
    axs.scatter(
        [], [],
        label=r'$\langle \mathrm{eff}(m_H)\rangle$' + f': {np.mean(m_H_efficiences):.3}',
        s=0.0
    )
    # axs.set_ylim([0.0, 1.0])
    axs.set_xlabel(r'$m_H$ input')
    axs.set_ylabel(r'$m_H$ efficiency')
    axs.set_title(r'CORENet $m_H$ efficiency vs. $m_H$')
    axs.legend()
    plt.savefig(f'{save_directory}/m_H_efficiency.png')
    plt.close()

    fig, axs = plt.subplots(figsize=(10, 6))
    dm_efficiencies = np.mean(efficiencies['dm_efficiency'], axis=1)
    axs.plot(
        efficiencies['y_centers'],
        dm_efficiencies,
        linestyle='--'
    )
    axs.scatter(
        [], [],
        label=r'$\langle \mathrm{eff}(\Omega_{DM})\rangle$' + f': {np.mean(dm_efficiencies):.3}',
        s=0.0
    )
    # axs.set_ylim([0.0, 1.0])
    axs.set_xlabel(r'$\Omega_{DM}$ input')
    axs.set_ylabel(r'$\Omega_{DM}$ efficiency')
    axs.set_title(r'CORENet $\Omega_{DM}$ efficiency vs. $\Omega_{DM}$')
    axs.legend()
    plt.savefig(f'{save_directory}/dm_efficiency.png')
    plt.close()
