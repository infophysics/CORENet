import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import argparse
from sklearn.neighbors import KernelDensity

from corenet.dataset.common import (
    cmssm_columns,
    pmssm_columns,
    common_columns,
    constraints_cmssm,
    constraints_pmssm
)


def import_data_efficient(data_dir, subspace):
    '''
    Method to read in the SUSY datasets while minimizing memory usage. Selects only datapoints that satisfy constraints.

    Arguments:
        data_dir: string, path to the folder containing the csv files.
        constraints: A function that selects points to retain from the CSV file after being read into Pandas data.
        subspace: 'cmssm' or'pmssm', specifies which dataset to read in.
    Returns:
        Numpy array containing the 'read_columns' entries in the dataset.
    '''
    # See the README file for a description of the file structure.
    if subspace == 'cmssm':
        all_columns = cmssm_columns + common_columns
        usecols = [0, 1, 2, 3, 4, 45, 50, 81, 91]
        constraints = constraints_cmssm
    if subspace == 'pmssm':
        all_columns = pmssm_columns + common_columns
        usecols = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19,
            45, 50, 81, 91
        ]
        constraints = constraints_pmssm
    filenames = [
        file for file in os.listdir(path=data_dir)
        if os.path.isfile(os.path.join(data_dir, file))
    ]
    individual_files = []
    for filename in filenames:
        file_dataframe = pd.read_csv(
            data_dir + "/" + filename,
            sep=',', header=None,
            names=all_columns,
            usecols=usecols
        )
        individual_files += [constraints(file_dataframe)]
    concatenated_files = pd.concat(individual_files)
    return concatenated_files.values


def boxcarpy(
    x,
    val=125.09,
    sigma=3.0,
    mode='reg'
):
    low = torch.tensor([val - sigma])
    hi = torch.tensor([val + sigma])
    if mode == 'reg':
        return -torch.heaviside(x-hi, torch.tensor([1.0])) + torch.heaviside(x-low, torch.tensor([1.0]))
    elif mode == 'anti':
        return torch.tensor([1.0]) + torch.heaviside(x-hi, torch.tensor([1.0])) - torch.heaviside(x-low, torch.tensor([1.0]))


def generate_dummy_points(
    input_file,
    data_type,
    validity,
    num_samples,
    kernel,
    bandwidth
):
    """Load in the input_file"""
    data = import_data_efficient(input_file, data_type)
    data = torch.from_numpy(data.astype('float32'))

    """Data type variables"""
    if data_type == "cmssm":
        index = 5
    else:
        index = 19

    """Get non zero points"""
    data = torch.squeeze(data[[boxcarpy(data[:, index]).nonzero()]])
    data = torch.squeeze(data[[
        boxcarpy(data[:, index+2], val=0.11, sigma=0.03).nonzero()
    ]])
    if validity == "triple":
        data = torch.squeeze(data[[(data[:, index+1] == data[:, index+3]).nonzero()]])

    minvec = data.detach().numpy().min(axis=0)
    maxvec = data.detach().numpy().max(axis=0)
    minvec[index] = 122.09
    minvec[index+2] = 0.08
    maxvec[index] = 128.09
    maxvec[index+2] = 0.14

    data = (data - minvec)/(maxvec - minvec)
    kde = KernelDensity(
        kernel=kernel,
        bandwidth=bandwidth
    ).fit(data[:, :index].detach().numpy())
    output = torch.tensor(kde.sample(n_samples=num_samples).astype('float32'))
    output = torch.hstack((output[:, :index], torch.zeros(num_samples, index), torch.rand(num_samples, 2))).numpy()
    np.savetxt('generated_points.txt', output, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CMSSM/PMSSM dummy data with corresponding uniform constraints"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file."
    )
    parser.add_argument(
        "data_type",
        type=str,
        help="Type of data being processed (e.g., 'cmssm', 'pmssm')."
    )
    parser.add_argument(
        "validity",
        type=str,
        choices=["double", "triple"],
        help="Specify whether the data is double or triple valid."
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of samples to generate."
    )
    parser.add_argument(
        "kernel",
        type=str,
        default="tophat",
        help="Kernel for sampling"
    )
    parser.add_argument(
        "bandwidth",
        type=float,
        default=0.1,
        help="Bandwidth value for kernel"
    )

    args = parser.parse_args()
    generate_dummy_points(
        args.input_file,
        args.data_type,
        args.validity,
        args.num_samples,
        args.kernel,
        args.bandwidth
    )
