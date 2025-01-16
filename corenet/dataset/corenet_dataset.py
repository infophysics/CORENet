

import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from corenet.utils.utils import generate_plot_grid
from corenet.utils.utils import fig_to_array


corenet_dataset_config = {
    "dataset_folder":   "data/",
    "dataset_files":    [""],
}


class CORENetDataset(Dataset):
    """
    """
    def __init__(
        self,
        name:   str = "corenet",
        config: dict = corenet_dataset_config,
        meta:   dict = {}
    ):
        self.name = name
        self.config = config
        self.meta = meta

        self.process_config()

    def process_config(self):
        self.num_events = 10

        self.dataset_folder = self.config['dataset_folder']
        self.files = [
            file for file in os.listdir(path=self.dataset_folder)
            if os.path.isfile(os.path.join(self.dataset_folder, file))
        ]
        data_list = []

        for file in self.files:
            # Read the data from each file and append it to the list
            try:
                data = np.loadtxt(self.dataset_folder + '/' + file, delimiter=',')  # Adjust delimiter if needed
                data_list.append(data)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Combine all data into a single numpy array
        self.data_set = np.vstack(data_list)
        # shuffle the dataset
        np.random.shuffle(self.data_set)
        self.num_events = len(self.data_set)
        self.gut_test = self.config['gut_test']
        self.gut_true = self.config['gut_true']
        self.weak_test = self.config['weak_test']

        if "normalized" not in self.config:
            self.config["normalized"] = False
        self.normalized = self.config["normalized"]

        if self.normalized:
            gut_test = torch.Tensor(self.data_set[:, self.gut_test[0]:self.gut_test[1]])
            gut_corr = torch.Tensor(self.data_set[:, self.gut_true[0]:self.gut_true[1]])
            weak_test = torch.Tensor(self.data_set[:, self.weak_test[0]:self.weak_test[1]])
            gut_true = gut_test - gut_corr
            gut = torch.cat((gut_test, gut_true))
            self.gut_means = torch.mean(gut, dim=0)
            self.gut_stds = torch.std(gut, dim=0)
            self.weak_means = torch.mean(weak_test, dim=0)
            self.weak_stds = torch.std(weak_test, dim=0)

        self.dataset_type = self.config['dataset_type']
        if self.dataset_type == 'cmssm':
            self.gut_variable_names = [r'$m_0$', r'$m_{1/2}$', r'$A_0$', r'$\mathrm{tan} \beta$', r'$\mathrm{sign}(\mu)$']
            self.gut_variable_file_names = ['m_0', 'm_12', 'A_0', 'tan_beta', 'sign_mu']
        else:
            self.gut_variable_names = ['']
        self.weak_variable_names = ['higgs', 'relic_dm']

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        gut_test = torch.Tensor(self.data_set[idx, self.gut_test[0]:self.gut_test[1]])
        gut_corr = torch.Tensor(self.data_set[idx, self.gut_true[0]:self.gut_true[1]])
        weak_test = torch.Tensor(self.data_set[idx, self.weak_test[0]:self.weak_test[1]])
        data = {
            'gut_test': gut_test,
            'gut_true': gut_test - gut_corr,
            'weak_test': weak_test
        }
        if self.normalized:
            data = self.normalize(data)
        return data

    def normalize(
        self,
        data
    ):
        data['gut_test'] = (data['gut_test'] - self.gut_means)/self.gut_stds
        data['gut_true'] = (data['gut_true'] - self.gut_means)/self.gut_stds
        data['weak_test'] = (data['weak_test'] - self.weak_means)/self.weak_stds
        return data

    def unnormalize(
        self,
        data,
    ):
        data['gut_test'] = self.gut_means + self.gut_stds * data['gut_test']
        data['gut_true'] = self.gut_means + self.gut_stds * data['gut_true']
        data['weak_test'] = self.weak_means + self.weak_stds * data['weak_test']
        if 'gut_test_output' in data.keys():
            data['gut_test_output'] = self.gut_means + self.gut_stds * data['gut_test_output']
        if 'gut_true_output' in data.keys():
            data['gut_true_output'] = self.gut_means + self.gut_stds * data['gut_true_output']
        if 'weak_test_output' in data.keys():
            data['weak_test_output'] = self.gut_means + self.gut_stds * data['weak_test_output']
        return data

    def save_predictions(
        self,
        model_name,
        predictions,
        indices
    ):
        pass

    def evaluate_outputs(
        self,
        data,
        data_type='training'
    ):
        """
        Here we make plots of the distributions of gut_test/gut_true before and after the autoencorder,
        as well as different plots of the latent projections, binary variables, etc.
        """
        if self.normalized:
            data = self.unnormalize(data)
        for ii, gut_variable in enumerate(self.gut_variable_names):
            fig, axs = plt.subplots()
            try:
                axs.hist(
                    data['gut_test'][:, ii].numpy(),
                    bins=25,
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.hist(
                    data['gut_test'][:, ii].numpy(),
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            try:
                axs.hist(
                    data['gut_test_output'][:, ii].numpy(),
                    bins=25,
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.hist(
                    data['gut_test_output'][:, ii].numpy(),
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            axs.set_xlabel(gut_variable)
            axs.legend()
            plt.suptitle(f"GUT test ({gut_variable}) - {data_type} input/output")
            plt.tight_layout()
            plt.savefig(f'{self.meta["plot_directory"]}/{self.gut_variable_file_names[ii]}_gut_test_{data_type}.png')
            fig_array = fig_to_array(fig)
            self.meta['tensorboard'].add_image(
                f'{self.gut_variable_file_names[ii]} (gut_test) {data_type}',
                fig_array,
                0,
                dataformats='HWC'
            )
            plt.close()
        """Make plot grid"""
        fig, axs = generate_plot_grid(num_plots=len(self.gut_variable_names), figsize=(10, 6))
        for ii, gut_variable in enumerate(self.gut_variable_names):
            try:
                axs.flat[ii].hist(
                    data['gut_test'][:, ii].numpy(),
                    bins=25,
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.flat[ii].hist(
                    data['gut_test'][:, ii].numpy(),
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            try:
                axs.flat[ii].hist(
                    data['gut_test_output'][:, ii].numpy(),
                    bins=25,
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.flat[ii].hist(
                    data['gut_test_output'][:, ii].numpy(),
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            axs.flat[ii].set_xlabel(gut_variable)
        axs.flat[0].legend()
        plt.suptitle(f"GUT test - {data_type} input/output")
        plt.tight_layout()
        plt.savefig(f'{self.meta["plot_directory"]}/gut_test_{data_type}.png')
        fig_array = fig_to_array(fig)
        self.meta['tensorboard'].add_image(
            f'(gut_test) {data_type}',
            fig_array,
            0,
            dataformats='HWC'
        )
        plt.close()

        """Same thing for GUT True"""
        for ii, gut_variable in enumerate(self.gut_variable_names):
            fig, axs = plt.subplots()
            try:
                axs.hist(
                    data['gut_true'][:, ii].numpy(),
                    bins=25,
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.hist(
                    data['gut_true'][:, ii].numpy(),
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            try:
                axs.hist(
                    data['gut_true_output'][:, ii].numpy(),
                    bins=25,
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.hist(
                    data['gut_true_output'][:, ii].numpy(),
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            axs.set_xlabel(gut_variable)
            axs.legend()
            plt.suptitle(f"GUT true ({gut_variable}) - {data_type} input/output")
            plt.tight_layout()
            plt.savefig(f'{self.meta["plot_directory"]}/{self.gut_variable_file_names[ii]}_gut_true_{data_type}.png')
            fig_array = fig_to_array(fig)
            self.meta['tensorboard'].add_image(
                f'{self.gut_variable_file_names[ii]} (gut_true) {data_type}',
                fig_array,
                0,
                dataformats='HWC'
            )
            plt.close()
        """Make plot grid"""
        fig, axs = generate_plot_grid(num_plots=len(self.gut_variable_names), figsize=(10, 6))
        for ii, gut_variable in enumerate(self.gut_variable_names):
            try:
                axs.flat[ii].hist(
                    data['gut_true'][:, ii].numpy(),
                    bins=25,
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.flat[ii].hist(
                    data['gut_true'][:, ii].numpy(),
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            try:
                axs.flat[ii].hist(
                    data['gut_true_output'][:, ii].numpy(),
                    bins=25,
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.flat[ii].hist(
                    data['gut_true_output'][:, ii].numpy(),
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            axs.flat[ii].set_xlabel(gut_variable)
        axs.flat[0].legend()
        plt.suptitle(f"GUT true - {data_type} input/output")
        plt.tight_layout()
        plt.savefig(f'{self.meta["plot_directory"]}/gut_true_{data_type}.png')
        fig_array = fig_to_array(fig)
        self.meta['tensorboard'].add_image(
            f'(gut_true) {data_type}',
            fig_array,
            0,
            dataformats='HWC'
        )
        plt.close()

        """Same thing for Combo"""
        for ii, gut_variable in enumerate(self.gut_variable_names):
            fig, axs = plt.subplots()
            try:
                axs.hist(
                    np.concatenate((data['gut_true'][:, ii].numpy(), data['gut_test'][:, ii].numpy())),
                    bins=25,
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.hist(
                    np.concatenate((data['gut_true'][:, ii].numpy(), data['gut_test'][:, ii].numpy())),
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            try:
                axs.hist(
                    np.concatenate((data['gut_true_output'][:, ii].numpy(), data['gut_test_output'][:, ii].numpy())),
                    bins=25,
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.hist(
                    np.concatenate((data['gut_true_output'][:, ii].numpy(), data['gut_test_output'][:, ii].numpy())),
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            axs.set_xlabel(gut_variable)
            axs.legend()
            plt.suptitle(f"GUT true/test ({gut_variable}) - {data_type} input/output")
            plt.tight_layout()
            plt.savefig(f'{self.meta["plot_directory"]}/{self.gut_variable_file_names[ii]}_gut_{data_type}.png')
            fig_array = fig_to_array(fig)
            self.meta['tensorboard'].add_image(
                f'{self.gut_variable_file_names[ii]} {data_type}',
                fig_array,
                0,
                dataformats='HWC'
            )
            plt.close()
        """Make plot grid"""
        fig, axs = generate_plot_grid(num_plots=len(self.gut_variable_names), figsize=(10, 6))
        for ii, gut_variable in enumerate(self.gut_variable_names):
            try:
                axs.flat[ii].hist(
                    np.concatenate((data['gut_true'][:, ii].numpy(), data['gut_test'][:, ii].numpy())),
                    bins=25,
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.flat[ii].hist(
                    np.concatenate((data['gut_true'][:, ii].numpy(), data['gut_test'][:, ii].numpy())),
                    label=f'input_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            try:
                axs.flat[ii].hist(
                    np.concatenate((data['gut_true_output'][:, ii].numpy(), data['gut_test_output'][:, ii].numpy())),
                    bins=25,
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            except Exception:
                axs.flat[ii].hist(
                    np.concatenate((data['gut_true_output'][:, ii].numpy(), data['gut_test_output'][:, ii].numpy())),
                    label=f'output_{data_type}',
                    histtype='step',
                    stacked=True,
                    density=True
                )
            axs.flat[ii].set_xlabel(gut_variable)
        axs.flat[0].legend()
        plt.suptitle(f"GUT true/test - {data_type} input/output")
        plt.tight_layout()
        plt.savefig(f'{self.meta["plot_directory"]}/gut_{data_type}.png')
        fig_array = fig_to_array(fig)
        self.meta['tensorboard'].add_image(
            f'{data_type}',
            fig_array,
            0,
            dataformats='HWC'
        )
        plt.close()
