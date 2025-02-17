import numpy as np
import torch
import os
import pandas as pd
import argparse
from sklearn.neighbors import KernelDensity, KDTree


def import_data_efficient(data_dir, subspace):
    """
    Reads the SUSY datasets while minimizing memory usage.
    Selects only datapoints that satisfy constraints.
    """
    filenames = [
        file for file in os.listdir(path=data_dir)
        if os.path.isfile(os.path.join(data_dir, file))
    ]
    individual_files = []
    for filename in filenames:
        file_dataframe = pd.read_csv(
            os.path.join(data_dir, filename), sep=',', header=None
        )
        individual_files.append(file_dataframe)
    concatenated_files = pd.concat(individual_files)
    return concatenated_files.values


def generate_dummy_points(
    input_file,
    data_type,
    validity,
    num_samples,
    kernel,
    bandwidth,
    output_folder,
    output_name,
):
    """Generates dummy GUT points and retrieves nearest weak values."""
    data = import_data_efficient(input_file, data_type)
    data = torch.from_numpy(data.astype('float32'))

    gut_values = data[:, :5]  # Extract GUT values (assuming first 5 columns are GUT)
    weak_values = data[:, 5:7]  # Extract WEAK values (assuming columns 5-7 are weak inputs)

    # Normalize data
    minvec = gut_values.min(dim=0)[0]
    maxvec = gut_values.max(dim=0)[0]
    gut_values = (gut_values - minvec) / (maxvec - minvec)

    # Train KDE on normalized GUT values
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(gut_values.numpy())
    sampled_gut = torch.tensor(kde.sample(n_samples=num_samples), dtype=torch.float32)

    # Find the closest GUT point in the dataset
    tree = KDTree(gut_values.numpy())
    _, nearest_idx = tree.query(sampled_gut.numpy())

    # Retrieve corresponding weak values from the closest training GUT points
    sampled_weak = weak_values[nearest_idx.flatten()]

    # Save the generated samples
    output = torch.hstack((sampled_gut, sampled_weak))
    np.savetxt(f'{output_folder}/{output_name}_generated_points.txt', output.numpy(), delimiter=",")
    np.savez(
        f'{output_folder}/{output_name}_generated_points_norm.npz',
        minvec=minvec.numpy(),
        maxvec=maxvec.numpy()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GUT samples with correct weak values")
    parser.add_argument("input_file", type=str, help="Path to the input dataset folder")
    parser.add_argument("data_type", type=str, help="Type of data being processed (e.g., 'cmssm', 'pmssm')")
    parser.add_argument("validity", type=str, choices=["double", "triple"], help="Specify validity filtering")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate")
    parser.add_argument("kernel", type=str, default="tophat", help="KDE kernel type")
    parser.add_argument("bandwidth", type=float, default=0.1, help="KDE bandwidth")
    parser.add_argument("output_folder", type=str, help="Path to save generated points")
    parser.add_argument("output_name", type=str, help="Output filename prefix")

    args = parser.parse_args()
    generate_dummy_points(
        args.input_file,
        args.data_type,
        args.validity,
        args.num_samples,
        args.kernel,
        args.bandwidth,
        args.output_folder,
        args.output_name
    )
