"""
"""
from pathlib import Path

from corenet.utils.logger import Logger
from corenet.utils.config import ConfigParser
from corenet.utils.utils import get_datetime
from corenet.module import ModuleHandler

import torch
import os
from torch.utils.tensorboard import SummaryWriter

os.environ["TQDM_NOTEBOOK"] = "false"


class CORENetRunner:
    """
    """
    def __init__(
        self,
        config_file:    str,
        run_name:       str = None,
        local_scratch:  str = './',
        local_corenet:     str = './',
        local_data:     str = './',
        anomaly:        bool = False,
        dataset_folder: str = './'
    ):
        # set up directories
        self.config_file = config_file
        self.run_name = run_name
        self.local_scratch = local_scratch
        self.local_corenet = local_corenet
        self.local_data = local_data
        self.anomaly = anomaly
        self.dataset_folder = dataset_folder

        if not os.path.isdir(self.local_scratch):
            self.local_scratch = './'
        if not os.path.isdir(self.local_corenet):
            self.local_corenet = './'
        if not os.path.isdir(self.local_data):
            self.local_data = './'
        self.local_corenet_files = [
            self.local_corenet + '/' + file
            for file in os.listdir(path=os.path.dirname(self.local_corenet))
        ]
        self.local_data_files = [
            self.local_data + '/' + file
            for file in os.listdir(path=os.path.dirname(self.local_data))
        ]
        os.environ['LOCAL_SCRATCH'] = self.local_scratch
        os.environ['LOCAL_CORENET'] = self.local_corenet
        os.environ['LOCAL_DATA'] = self.local_data

        self.logger = Logger('corenet_runner')

        # begin parsing configuration file
        if self.config_file is None:
            self.logger.error('no config_file specified in parameters!')

        self.config = ConfigParser(self.config_file).data

        if self.anomaly:
            self.logger.info(f'setting anomaly detection to {self.anomaly}')
            torch.autograd.set_detect_anomaly(bool(self.anomaly))

        if "module" not in self.config.keys():
            self.logger.error('"module" section not specified in config!')

        self.set_up_meta()
        self.set_up_devices()
        self.set_up_modules()

    def set_up_meta(self):
        self.logger.info("configuring meta...")

        system_info = self.logger.get_system_info()
        for key, value in system_info.items():
            self.logger.info(f"system_info - {key}: {value}")

        # get run_name
        if self.run_name is None:
            self.run_name = self.config['module']['module_name']

        # add unique datetime
        now = get_datetime()
        self.run_name += f"_{now}"
        if "local_run" in self.config['module']:
            self.local_run = self.config['module']['local_run']
        else:
            self.local_run = self.local_scratch + '/' + self.run_name

        # create run directory
        if not os.path.isdir(self.local_run):
            os.makedirs(self.local_run)

        # create plots directory
        self.local_plots = self.local_run + '/plots/'
        if not os.path.isdir(self.local_plots):
            os.makedirs(self.local_plots)

        self.meta = {
            'system_info':      system_info,
            'now':              now,
            'run_name':         self.run_name,
            'config_file':      self.config_file,
            'run_directory':    self.local_run,
            'plot_directory':   self.local_plots,
            'local_scratch':    self.local_scratch,
            'local_corenet':       self.local_corenet,
            'local_data':       self.local_data,
            'local_corenet_files': self.local_corenet_files,
            'local_data_files': self.local_data_files
        }

        # Take inference config and copy it to the run directory with the right edits
        if self.config["module"]["module_mode"][0] == "training":
            config_parser = ConfigParser(self.local_corenet + '/config/inference.yaml')
            inference_config = config_parser.data
            inference_config["module"]["local_run"] = self.local_run
            inference_config["dataset"]["norm_params"] = self.local_run + '/norm_params.npz'
            inference_config["model"]["load_model"] = self.local_run + '/CORENet_trained_params.ckpt'
            inference_config["model"]["model_config"] = self.local_run + '/CORENet_init.config'
            config_parser.save_config(inference_config, self.local_run + '/inference.yaml')
        else:
            config_parser = ConfigParser(self.local_corenet + '/config/mssm.yaml')
            mssm_config = config_parser.data
            mssm_config["mssm"]["mssm_folder"] = self.local_run
            mssm_config["mssm"]["mssm_files"][0] = self.local_run + self.run_name + ".npz"
            config_parser.save_config(mssm_config, self.local_run + '/mssm.yaml')

        if self.dataset_folder != "":
            self.config["dataset"]["dataset_folder"] = self.dataset_folder

        # set up tensorboard
        self.meta['tensorboard_dir'] = self.meta['run_directory']
        self.meta['tensorboard'] = SummaryWriter(
            log_dir=self.meta['tensorboard_dir']
        )
        self.logger.info(f'"now" set to: {now}')
        self.logger.info(f'"run_name" set to: {self.run_name}')
        self.logger.info(f'"run_directory" set to: {self.local_run}')
        self.logger.info(f'"local_scratch" directory set to: {self.local_scratch}')
        self.logger.info(f'"local_corenet" directory set to: {self.local_corenet}')
        self.logger.info(f'"local_data" directory set to: {self.local_data}')

        # set verbosity of self.logger
        if "verbose" in self.config["module"]:
            if not isinstance(self.config["module"]["verbose"], bool):
                self.logger.error(f'"module:verbose" must be of type bool, but got {type(self.config["module"]["verbose"])}!')
            self.meta["verbose"] = self.config["module"]["verbose"]
        else:
            self.meta["verbose"] = False

    def set_up_devices(self):
        # check for devices
        if "gpu" not in self.config["module"].keys():
            self.logger.warn('"module:gpu" not specified in config!')
            gpu = None
        else:
            gpu = self.config["module"]["gpu"]
        if "gpu_device" not in self.config["module"].keys():
            self.logger.warn('"module:gpu_device" not specified in config!')
            gpu_device = None
        else:
            gpu_device = self.config["module"]["gpu_device"]

        if torch.cuda.is_available():
            self.logger.info("CUDA is available with devices:")
            for ii in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(ii)
                cuda_stats = f"name: {device_properties.name}, "
                cuda_stats += f"compute: {device_properties.major}.{device_properties.minor}, "
                cuda_stats += f"memory: {device_properties.total_memory}"
                self.logger.info(f" -- device: {ii} - " + cuda_stats)

        # set gpu settings
        if gpu:
            if torch.cuda.is_available():
                if gpu_device >= torch.cuda.device_count() or gpu_device < 0:
                    self.logger.warn(f"desired gpu_device '{gpu_device}' not available, using device '0'")
                    gpu_device = 0
                self.meta['device'] = torch.device(f"cuda:{gpu_device}")
                self.logger.info(
                    f"CUDA is available, using device {gpu_device}" +
                    f": {torch.cuda.get_device_name(gpu_device)}"
                )
            else:
                gpu = False
                self.logger.warn("CUDA not available! Using the cpu")
                self.meta['device'] = torch.device("cpu")
        else:
            self.logger.info("using cpu as device")
            self.meta['device'] = torch.device("cpu")
        self.meta['gpu'] = gpu

    def set_up_modules(self):
        # Configure the module handler
        self.logger.info("configuring modules.")
        module_config = self.config
        self.module_handler = ModuleHandler(
            self.run_name,
            module_config,
            meta=self.meta
        )

    def get_products(self):
        return self.meta, self.module_handler


def parse_command_line_config(
    params
):
    # set up local scratch directory
    if params.local_scratch is not None:
        if not os.path.isdir(params.local_scratch):
            if "LOCAL_SCRATCH" in os.environ:
                params.local_scratch = os.environ["LOCAL_SCRATCH"]
            else:
                params.local_scratch = './'
    else:
        if "LOCAL_SCRATCH" in os.environ:
            params.local_scratch = os.environ["LOCAL_SCRATCH"]
        else:
            params.local_scratch = './'

    # set up local corenet directory
    if params.local_corenet is not None:
        if not os.path.isdir(params.local_corenet):
            if "LOCAL_CORENET" in os.environ:
                params.local_corenet = os.environ["LOCAL_CORENET"]
            else:
                params.local_corenet = str(Path(__file__).resolve().parent)

    else:
        if "LOCAL_CORENET" in os.environ:
            params.local_corenet = os.environ["LOCAL_CORENET"]
        else:
            params.local_corenet = str(Path(__file__).resolve().parent)

    # set up local data directory
    if params.local_data is not None:
        if not os.path.isdir(params.local_data):
            if "LOCAL_DATA" in os.environ:
                params.local_data = os.environ["LOCAL_DATA"]
            else:
                params.local_data = './'
    else:
        if "LOCAL_DATA" in os.environ:
            params.local_data = os.environ["LOCAL_DATA"]
        else:
            params.local_data = './'

    corenet_runner = CORENetRunner(
        params.config_file,
        params.name,
        params.local_scratch,
        params.local_corenet,
        params.local_data,
        params.anomaly,
        params.dataset_folder
    )
    return corenet_runner.get_products()
