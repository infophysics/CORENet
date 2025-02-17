"""
"""
from mpi4py import MPI
import os
import sys
import glob
from tqdm import tqdm
import torch
import csv
import numpy as np
from datetime import datetime
import traceback
from importlib.metadata import version

from corenet.utils.logger import Logger
from corenet.programs.common import (
    sugra_input,
    universal_input
)
from corenet.dataset.common import (
    softsusy_physical_parameters,
    softsusy_weak_parameters,
    higgs_mass,
    higgs_mass_sigma,
    dm_relic_density,
    dm_relic_density_sigma
)


class MSSM:
    """
    Main MSSM class for running jobs. MSSM is designed
    to work with MPI and H5.  The user must run MSSM using mpirun
    with at least two processes (one master and N-1 workers).
    """
    def __init__(
        self,
        config: dict = {},
        meta:   dict = {},
    ):
        """_summary_

        Args:
            config (dict): config file for running MSSM.
            meta (dict): dictionary of meta information to
            be shared across all nodes.
        """

        """Get mpi communication parameters"""
        try:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        except Exception as exception:
            raise RuntimeError(f"unable to obtain MPI parameters: {exception}")

        """Set input parameters and set up loggers"""
        try:
            if self.rank == 0:
                self.logger = Logger(f"master: {self.rank}")
            else:
                self.logger = Logger(f"worker: {self.rank}")
        except Exception as exception:
            raise RuntimeError(f"unable to set up logging system: {exception}")

        self.config = config
        self.meta = meta

        """Setting error status for this node"""
        self.error_status = None
        self.exc_type = None
        self.exc_value = None
        self.exc_traceback = None
        self.line_number = None
        self.file_name = None
        self.tb_str = None
        self.traceback_details = None
        self.event_errors = []
        self.plugin_errors = []
        self.event_plugin_errors = []
        self.event_plugin_exc_types = []
        self.event_plugin_exc_values = []
        self.event_plugin_exc_tracebacks = []
        self.event_plugin_line_numbers = []
        self.event_plugin_file_names = []
        self.event_plugin_tb_strs = []
        self.event_plugin_traceback_details = []

        """Parse config"""
        try:
            self.parse_config()
        except Exception as exception:
            self.report_error(exception=exception)
        self.barrier()

        self.mssm_files = []
        """Distributed events and indices for various arrays"""
        self.distributed_events = {}

    def report_error(
        self,
        exception: Exception = None
    ):
        self.error_status = exception
        self.exc_type, self.exc_value, self.exc_traceback = sys.exc_info()
        # Extracting the line number from the traceback
        self.line_number = self.exc_traceback.tb_lineno
        self.file_name = self.exc_traceback.tb_frame.f_code.co_filename
        # Optionally, use traceback to format a string of the entire traceback
        self.tb_str = traceback.format_exception(self.exc_type, self.exc_value, self.exc_traceback)
        self.traceback_details = "".join(self.tb_str)

    def barrier(self):
        errors = self.comm.allgather(self.error_status)
        exc_types = self.comm.allgather(self.exc_type)
        line_numbers = self.comm.allgather(self.line_number)
        file_names = self.comm.allgather(self.file_name)
        traceback_details = self.comm.allgather(self.traceback_details)
        if any(errors):
            if self.rank == 0:
                errors_count = sum(1 for error in errors if error is not None)
                errors_indices = [index for index, error in enumerate(errors) if error is not None]
                self.logger.critical(f"{errors_count} errors encountered in MSSM program")
                for index in errors_indices:
                    self.logger.critical(f"error encountered in worker {index}: ")
                    self.logger.critical(f"exception:   {errors[index]}")
                    self.logger.critical(f"exc_type:    {exc_types[index]}")
                    self.logger.critical(f"line_number:     {line_numbers[index]}")
                    self.logger.critical(f"file_name:       {file_names[index]}")
                    self.logger.critical(f"traceback:   {traceback_details[index]}")
                self.comm.Abort(1)
        else:
            self.comm.Barrier()

    def parse_config(self):
        """
        Set up config parameters from input config file
        """
        self.param_space = self.config["mssm"]["param_space"]
        self.softsusy_dir = self.config["mssm"]["softsusy_dir"]
        self.micromegas_dir = self.config["mssm"]["micromegas_dir"]
        self.softsusy_cmd = self.softsusy_dir + "/softpoint.x leshouches < "
        self.micromegas_cmd = self.micromegas_dir + "/main "
        self.normalization_file = self.config["mssm"]["normalization_file"]
        norm_file = np.load(self.normalization_file)
        self.minvec = norm_file['minvec']
        self.maxvec = norm_file['maxvec']

        if self.rank == 0:
            try:
                """Startup main mssm program"""
                self.logger.info(
                    f'############################ MSSM  v. [{version("corenet")}] ############################'
                )

                """Check for main mssm parameters"""
                if "mssm" not in self.config.keys():
                    self.logger.error("mssm section not in config!")
                mssm_config = self.config["mssm"]

                """Try to grab system info and display to the logger"""
                system_info = self.logger.get_system_info()
                time = datetime.now()
                now = f"{time.hour}:{time.minute}:{time.second} [{time.day}/{time.month}/{time.year}]"
                self.logger.info(f'system_info - local time: {now}')
                for key, value in system_info.items():
                    self.logger.info(f"system_info - {key}: {value}")

                """Set the verbosity of the mssm program"""
                if "verbose" in mssm_config:
                    if not isinstance(mssm_config["verbose"], bool):
                        self.logger.error(
                            f'"mssm:verbose" must be of type bool, but got {type(mssm_config["verbose"])}!'
                        )
                    self.meta["verbose"] = mssm_config["verbose"]
                else:
                    self.meta["verbose"] = False

                """See if any CUDA devices are available on the system"""
                if torch.cuda.is_available():
                    self.logger.info("CUDA is available with devices:")
                    for ii in range(torch.cuda.device_count()):
                        device_properties = torch.cuda.get_device_properties(ii)
                        self.logger.info(f" -- device: {ii}")
                        self.logger.info(f" {' ':<{5}} name: {device_properties.name}")
                        self.logger.info(f" {' ':<{5}} compute: {device_properties.major}.{device_properties.minor}")
                        self.logger.info(f" {' ':<{5}} memory: {round(device_properties.total_memory / (1024**3))} GB")

                """Send new device meta data to workers"""
                for ii in range(1, self.comm.size):
                    self.comm.send(self.meta, dest=ii, tag=0)
            except Exception as exception:
                self.report_error(exception=exception)
            self.barrier()
        else:
            self.barrier()
            try:
                self.meta = self.comm.recv(source=0, tag=0)
            except Exception as exception:
                self.report_error(exception=exception)

    def set_up_input_files(self):
        """
        Iterate over mssm_folder directory and determine
        if listed input files exist, or if "all" is selected for
        input files, construct the list of all .h5 files in the
        mssm_folder.

        We create the data members
            self.mssm_folder - location of the mssm files specified in config
            self.mssm_files - names of all the mssm files to process
            self.mssm_folder - location to put the mssm files
        """
        if self.rank == 0:
            try:
                mssm_dict = self.config['mssm']

                """Check for parameters"""
                if 'mssm_folder' not in mssm_dict.keys():
                    self.logger.error('mssm_folder not specified in config!')
                if 'mssm_files' not in mssm_dict.keys():
                    self.logger.error('mssm_files not specified in config!')

                mssm_folder = mssm_dict['mssm_folder']
                mssm_files = mssm_dict["mssm_files"]

                """Check that mssm folder exists"""
                if not os.path.isdir(mssm_folder):
                    self.logger.error(f'specified mssm_folder {mssm_folder} does not exist!')

                """Check that mssm folder has a '/' at the end"""
                if mssm_folder[-1] != '/':
                    mssm_folder += '/'

                """Check that mssm folder has a '/' at the end"""
                if mssm_folder[-1] != '/':
                    mssm_folder += '/'

                if isinstance(mssm_dict["mssm_files"], list):
                    """
                    If the mssm_files parameter is a list, look through
                    the list and make sure each specified file actually exists
                    in the mssm_folder.
                    """
                    mssm_files = [
                        input_file for input_file in mssm_dict["mssm_files"]
                        if input_file not in mssm_dict["skip_files"]
                    ]
                    for mssm_file in mssm_files:
                        if not os.path.isfile(mssm_folder + mssm_file):
                            self.logger.error(
                                f"specified file {mssm_file} does not exist in directory {mssm_folder}!"
                            )
                elif isinstance(mssm_dict["mssm_files"], str):
                    """
                    If the mssm_files parameter is a string, check if its
                    the phrase 'all', and if so, recursively grab all h5
                    files in the mssm_folder.

                    Otherwise, assume that the mssm_files parameter is a
                    file extension, and search recursively for all files
                    with that extension.
                    """
                    try:
                        self.logger.info(
                            f'searching {mssm_folder} recursively for all {mssm_dict["mssm_files"]} files.'
                        )
                        mssm_files = [
                            os.path.basename(input_file) for input_file in glob.glob(
                                f'{mssm_folder}/*.{mssm_dict["mssm_files"]}',
                                recursive=True,
                            )
                            if input_file not in mssm_dict["skip_files"]
                        ]
                    except Exception as exception:
                        self.logger.error(
                            f'specified "mssm_files" parameter: {mssm_dict["mssm_files"]} incompatible!'
                            + f" exception: {exception}"
                        )
                else:
                    self.logger.error(
                        f'specified "mssm_files" parameter: {mssm_dict["mssm_files"]} incompatible!'
                    )
                self.mssm_folder = mssm_folder
                self.mssm_files = mssm_files
                self.logger.info(f'setting mssm_folder to {self.mssm_folder}.')
                for ii in range(1, self.comm.size):
                    self.comm.send(self.mssm_folder, dest=ii, tag=80)
                    self.comm.send(self.mssm_files, dest=ii, tag=81)
                self.logger.info(f'found {len(self.mssm_files)} mssm files for processing.')
            except Exception as exception:
                self.report_error(exception=exception)
            self.barrier()
        else:
            self.barrier()
            try:
                self.mssm_folder = self.comm.recv(source=0, tag=80)
                self.mssm_files = self.comm.recv(source=0, tag=81)
            except Exception as exception:
                self.report_error(exception=exception)

    def reset_event_errors(self):
        pass

    def report_event_errors(
        self,
        file_name: str
    ):
        """
        Report any event/plugin errors that have occurred during
        this file.  These errors, if there are any, are reported
        and saved to the corresponding mssm file in the meta
        information.
        """
        if self.rank == 0:
            pass
        else:
            pass

    def parse_slha(
        self,
        event_id: int,
    ):
        """
        This function parses SLHA output files from SoftSUSY
        to get various parameter values.
        """
        parameters = {}
        input_values = []
        with open(f".tmp/susy_output_{event_id}", "r") as file:
            reader = csv.reader(file, delimiter='@')
            for row in reader:
                input_values.append([item for item in row])
        # loop through input values and look for blocks
        temp_block = ''
        for row in input_values:
            split_row = row[0].split()
            """
            First check if this line defines a new 'Block',
            and whether that block has an associated 'Q' value.
            """
            if split_row[0] == 'Block':
                temp_block = split_row[1]
                if temp_block not in parameters.keys():
                    parameters[temp_block] = {}
                if split_row[2] == 'Q=':
                    if 'Q' not in parameters[temp_block]:
                        parameters[temp_block]['Q'] = [round(float(split_row[3]), 6)]
                    else:
                        parameters[temp_block]['Q'].append(round(float(split_row[3]), 6))
                continue
            # if a comment line, then skip
            elif split_row[0] == '#':
                continue
            # Now parse the results of this block.
            elif split_row[2] == '#':
                row_type = split_row[3].replace('(Q)', '').replace('MSSM', '').replace('(MX)', '')
                if 'Q' in parameters[temp_block]:
                    if row_type not in parameters[temp_block]:
                        parameters[temp_block][row_type] = [round(float(split_row[1]), 6)]
                    else:
                        parameters[temp_block][row_type].append(round(float(split_row[1]), 6))
                else:
                    try:
                        parameters[temp_block][row_type] = [round(float(split_row[1]), 6)]
                    except Exception:
                        parameters[temp_block][row_type] = [split_row[1]]
            elif split_row[3] == '#':
                row_type = split_row[4].replace('(Q)', '').replace('MSSM', '').replace('(MX)', '')
                if 'Q' in parameters[temp_block]:
                    if row_type not in parameters[temp_block]:
                        parameters[temp_block][row_type] = [round(float(split_row[2]), 6)]
                    else:
                        parameters[temp_block][row_type].append(round(float(split_row[2]), 6))
                else:
                    try:
                        parameters[temp_block][row_type] = [round(float(split_row[2]), 6)]
                    except Exception:
                        parameters[temp_block][row_type] = [split_row[2]]
        return parameters

    def parse_micromegas(
        self,
        event_id: int,
    ):
        """
        This function parses the micromegas for a particular
        output slha file from softsusy.
        """
        # read in results
        with open(f".tmp/micrOmegas_output_{event_id}.txt", "r") as file:
            reader = csv.reader(file, delimiter=",")
            return next(reader)

    def compute_gut_coupling(
        self,
        parameters: dict
    ):
        """
        This function receives input from 'parse_slha' and
        computes various gauge couplings via linear interpolation.
        """
        gauge_couplings = [-1, -1, -1, -1]
        if self.param_space == 'pmssm':
            mx_scale = parameters['EXTPAR']['Set'][0]
        else:
            mx_scale = parameters['EXTPAR']['MX'][0]
        gauge_q = parameters['gauge']['Q']
        gauge_gprime = parameters['gauge']["g'"]
        gauge_g = parameters['gauge']['g']
        gauge_g3 = parameters['gauge']['g3']
        for ii in range(len(gauge_q)-1):
            if (gauge_q[ii] < mx_scale and gauge_q[ii+1] > mx_scale):
                high_q = gauge_q[ii+1]
                low_g,  high_g = gauge_g[ii], gauge_g[ii+1]
                low_g3, high_g3 = gauge_g3[ii], gauge_g3[ii+1]
                low_gprime, high_gprime = gauge_gprime[ii], gauge_gprime[ii+1]
                ratio = np.log10(mx_scale/high_q)
                # set gauge couplings
                gauge_couplings[0] = mx_scale
                gauge_couplings[1] = np.sqrt(5.0/3.0) * (low_gprime + ratio * (high_gprime - low_gprime))
                gauge_couplings[2] = low_g + ratio * (high_g - low_g)
                gauge_couplings[3] = low_g3 + ratio * (high_g3 - low_g3)
        return gauge_couplings

    def run_begin_of_file(
        self,
        file_name: str
    ):
        """
        Plugins to be run at the beginning of the file,
        which act on the entire file.

        Args:
            file_name (_str_): name of the input mssm file
        """
        if not os.path.isdir('.tmp/'):
            os.mkdir('.tmp/')

    def set_up_output_arrays(
        self,
        file_name: str
    ):
        """
        Construct output arrays for various data objects.
        This will take the mssm file and create a new .h5 file
        replacing the words "mssm" with "mssm" and "mssm"
        with "mssm".

        Args:
            file_name (str): name of the input mssm file
        """

        """Determine the mssm file name"""
        pass

    def distribute_tasks(self, file_name: str):
        """
        Distribute the entries of the 'gut_test', 'gut_test_output', and 'weak_test'
        arrays from the .npz file among the workers, ensuring that indices can be
        recombined later.
        Args:
            file_name (str): Path to the .npz file.
        """
        if self.rank == 0:
            try:
                # Master reads the .npz file
                data = np.load(self.mssm_folder + file_name)

                # Extract arrays
                gut_test = data['gut_test']
                gut_test_output = data['gut_test_output']
                weak_test = data['weak_test']

                # Determine the number of entries
                num_entries = len(gut_test)
                self.num_events = num_entries

                # Calculate how many entries each worker should handle
                chunk_size = num_entries // (self.size - 1)  # Exclude master
                remainder = num_entries % (self.size - 1)

                # Distribute indices to workers
                indices = []
                start = 0
                for i in range(1, self.size):  # Start with worker rank 1
                    end = start + chunk_size + (1 if i <= remainder else 0)
                    indices.append(list(range(start, end)))
                    start = end

                # Send the data to each worker
                for i in range(1, self.size):
                    self.comm.send({
                        'gut_test': gut_test[indices[i-1]],
                        'gut_test_output': gut_test_output[indices[i-1]],
                        'weak_test': weak_test[indices[i-1]],
                        'indices': indices[i-1]
                    }, dest=i, tag=10)

                # Log the distribution
                self.logger.info(f"Distributed {num_entries} entries across {self.size - 1} workers.")
            except Exception as exception:
                self.report_error(exception=exception)
        else:
            # Worker nodes receive their assigned data
            try:
                distributed_data = self.comm.recv(source=0, tag=10)
                self.distributed_events = distributed_data
                self.logger.info(f"Worker {self.rank} received {len(distributed_data['indices'])} entries.")
            except Exception as exception:
                self.report_error(exception=exception)

        self.barrier()

    def process_events_master(self):
        """
        Special code run by the master node on an event,
        which occurs before any of the workers.  Any work done
        here should be by plugins which create data that is
        needed by the other plugins.

        Args:
            mssm_file (_h5py.File_): input mssm_file
        """
        """Clear event indices so that file closes properly!"""
        pass

    def process_events_worker(self):
        """
        This function loops over all plugins and hands off the mssm_file,
        mssm_file and the associated indices for each that correspond
        to this workers event.

        Args:
            mssm_file (_h5py.File_): input mssm_file
        """
        gut_test = self.distributed_events['gut_test']
        gut_test_output = self.distributed_events['gut_test_output']
        indices = self.distributed_events['indices']
        gut_test_susy_physical_params = []
        gut_test_susy_weak_params = []
        gut_test_micromegas_params = []
        gut_test_gut_params = []
        gut_test_constraints = []
        gut_test_output_susy_physical_params = []
        gut_test_output_susy_weak_params = []
        gut_test_output_micromegas_params = []
        gut_test_output_gut_params = []
        gut_test_output_constraints = []
        super_invalids = 0

        self.progress_bar = tqdm(
            total=len(gut_test_output),
            position=self.rank,
            ncols=100,
            colour='MAGENTA',
            leave=True,
        )
        for ii, value in enumerate(gut_test_output):
            """Gut test output"""
            """Run SoftSUSY"""
            softsusy_error = self.run_softsusy(indices[ii], value)
            if (softsusy_error != 0):
                os.remove(f".tmp/input_{indices[ii]}.csv")
                os.remove(f".tmp/susy_output_{indices[ii]}")
                with open(".tmp/super_invalid_ids.txt", "a") as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow([indices[ii]])
                gut_test_output_susy_physical_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_physical_parameters.items())
                ])
                gut_test_output_susy_weak_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
                ])
                gut_test_output_micromegas_params.append([-1 for _ in range(66)])
                gut_test_output_gut_params.append([-1, -1, -1, -1])
                gut_test_output_constraints.append([-1, -1, -1])
                super_invalids += 1
                self.progress_bar.update(1)
                self.progress_bar.set_description(
                    f"worker (gut_test_output): {self.rank}"
                )
                self.progress_bar.set_postfix_str(f"# Super Invalid: {super_invalids}")
                continue
            """Run MicrOmegas"""
            micromegas_error = self.run_micromegas(indices[ii])
            if (micromegas_error != 0):
                os.remove(f".tmp/input_{indices[ii]}.csv")
                os.remove(f".tmp/susy_output_{indices[ii]}")
                with open(".tmp/super_invalid_ids.txt", "a") as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow([indices[ii]])
                gut_test_output_susy_physical_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_physical_parameters.items())
                ])
                gut_test_output_susy_weak_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
                ])
                gut_test_output_micromegas_params.append([-1 for _ in range(66)])
                gut_test_output_gut_params.append([-1, -1, -1, -1])
                gut_test_output_constraints.append([-1, -1, -1])
                super_invalids += 1
                self.progress_bar.update(1)
                self.progress_bar.set_description(
                    f"worker (gut_test_output): {self.rank}"
                )
                self.progress_bar.set_postfix_str(f"# Super Invalid: {super_invalids}")
                continue
            susy_params = self.parse_slha(indices[ii])
            gut_test_output_susy_physical_params.append([
                susy_params[item][key][0]
                for ii, (key, item) in enumerate(softsusy_physical_parameters.items())
            ])
            gut_test_output_susy_weak_params.append([
                susy_params[item][key][0]
                for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
            ])
            gut_test_output_micromegas_params.append(self.parse_micromegas(indices[ii]))
            gut_test_output_gut_params.append(self.compute_gut_coupling(susy_params))
            """Compute validities"""
            higgs_constraint = 0
            dm_constraint = 0
            lsp_constraint = 0
            if (
                abs(round(gut_test_output_susy_weak_params[-1][1], 2) - higgs_mass) < higgs_mass_sigma
            ):
                higgs_constraint = 1
            if (
                abs(round(float(gut_test_output_micromegas_params[-1][0]), 2) - dm_relic_density) < dm_relic_density_sigma
            ):
                dm_constraint = 1
            if (gut_test_output_susy_weak_params[-1][6] == float(gut_test_output_micromegas_params[-1][10])):
                lsp_constraint = 1
            gut_test_output_constraints.append([higgs_constraint, dm_constraint, lsp_constraint])
            os.remove(f".tmp/input_{indices[ii]}.csv")
            os.remove(f".tmp/susy_output_{indices[ii]}")
            self.progress_bar.update(1)
            self.progress_bar.set_description(
                f"worker (gut_test_output): {self.rank}"
            )
            self.progress_bar.set_postfix_str(f"# Super Invalid: {super_invalids}")
        self.progress_bar = tqdm(
            total=len(gut_test),
            position=self.rank,
            ncols=100,
            colour='MAGENTA',
            leave=True,
        )
        for ii, value in enumerate(gut_test):
            """GUT test"""
            softsusy_error = self.run_softsusy(indices[ii], value)
            if (softsusy_error != 0):
                os.remove(f".tmp/input_{indices[ii]}.csv")
                os.remove(f".tmp/susy_output_{indices[ii]}")
                with open(".tmp/super_invalid_ids.txt", "a") as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow([indices[ii]])
                gut_test_susy_physical_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_physical_parameters.items())
                ])
                gut_test_susy_weak_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
                ])
                gut_test_micromegas_params.append([-1 for _ in range(66)])
                gut_test_gut_params.append([-1, -1, -1, -1])
                gut_test_constraints.append([-1, -1, -1])
                super_invalids += 1
                self.progress_bar.update(1)
                self.progress_bar.set_description(
                    f"worker (gut_test): {self.rank}"
                )
                self.progress_bar.set_postfix_str(f"# Super Invalid: {super_invalids}")
                continue
            """Run MicrOmegas"""
            micromegas_error = self.run_micromegas(indices[ii])
            if (micromegas_error != 0):
                os.remove(f".tmp/input_{indices[ii]}.csv")
                os.remove(f".tmp/susy_output_{indices[ii]}")
                with open(".tmp/super_invalid_ids.txt", "a") as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow([indices[ii]])
                gut_test_susy_physical_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_physical_parameters.items())
                ])
                gut_test_susy_weak_params.append([
                    -1 for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
                ])
                gut_test_micromegas_params.append([-1 for _ in range(66)])
                gut_test_gut_params.append([-1, -1, -1, -1])
                gut_test_constraints.append([-1, -1, -1])
                super_invalids += 1
                self.progress_bar.update(1)
                self.progress_bar.set_description(
                    f"worker (gut_test): {self.rank}"
                )
                self.progress_bar.set_postfix_str(f"# Super Invalid: {super_invalids}")
                continue
            susy_params = self.parse_slha(indices[ii])
            gut_test_susy_physical_params.append([
                susy_params[item][key][0]
                for ii, (key, item) in enumerate(softsusy_physical_parameters.items())
            ])
            gut_test_susy_weak_params.append([
                susy_params[item][key][0]
                for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
            ])
            gut_test_micromegas_params.append(self.parse_micromegas(indices[ii]))
            gut_test_gut_params.append(self.compute_gut_coupling(susy_params))
            """Compute validities"""
            higgs_constraint = 0
            dm_constraint = 0
            lsp_constraint = 0
            if (
                abs(round(gut_test_susy_weak_params[-1][1], 2) - higgs_mass) < higgs_mass_sigma
            ):
                higgs_constraint = 1
            if (
                abs(round(float(gut_test_micromegas_params[-1][0]), 2) - dm_relic_density) < dm_relic_density_sigma
            ):
                dm_constraint = 1
            if (gut_test_susy_weak_params[-1][6] == float(gut_test_micromegas_params[-1][10])):
                lsp_constraint = 1
            gut_test_constraints.append([higgs_constraint, dm_constraint, lsp_constraint])
            os.remove(f".tmp/input_{indices[ii]}.csv")
            os.remove(f".tmp/susy_output_{indices[ii]}")
            self.progress_bar.update(1)
            self.progress_bar.set_description(
                f"worker (gut_test): {self.rank}"
            )
            self.progress_bar.set_postfix_str(f"# Super Invalid: {super_invalids}")
        self.gut_test_output_susy_physical_params = gut_test_output_susy_physical_params
        self.gut_test_output_susy_weak_params = gut_test_output_susy_weak_params
        self.gut_test_output_micromegas_params = gut_test_output_micromegas_params
        self.gut_test_output_gut_params = gut_test_output_gut_params
        self.gut_test_output_constraints = gut_test_output_constraints
        self.gut_test_susy_physical_params = gut_test_susy_physical_params
        self.gut_test_susy_weak_params = gut_test_susy_weak_params
        self.gut_test_micromegas_params = gut_test_micromegas_params
        self.gut_test_gut_params = gut_test_gut_params
        self.gut_test_constraints = gut_test_constraints

    def run_softsusy(
        self,
        event_id,
        gut_input
    ):
        """
        This function runs softsusy for a particular event
        input which is identified by its event id.
        """
        # create the input file
        if self.param_space == 'cmssm':
            gut_input = gut_input * (self.maxvec[:5] - self.minvec[:5]) + self.minvec[:5]
            negative_mu = (gut_input[:, 4] < 0.0)
            gut_input[:, 4][negative_mu] = -1.0
            gut_input[:, 4][~negative_mu] = 1.0
            tmp_input = sugra_input(
                m_scalar=gut_input[0],
                m_gaugino=gut_input[1],
                trilinear=gut_input[2],
                higgs_tanbeta=gut_input[3],
                sign_mu=gut_input[4]
            )
        else:
            gut_input = gut_input * (self.maxvec[:19] - self.minvec[:19]) + self.minvec[:19]
            tmp_input = universal_input(
                m_bino=gut_input[0],
                m_wino=gut_input[1],
                m_gluino=gut_input[2],
                trilinear_top=gut_input[3],
                trilinear_bottom=gut_input[4],
                trilinear_tau=gut_input[5],
                higgs_mu=gut_input[6],
                higgs_pseudo=gut_input[7],
                m_left_electron=gut_input[8],
                m_left_tau=gut_input[9],
                m_right_electron=gut_input[10],
                m_right_tau=gut_input[11],
                m_scalar_quark1=gut_input[12],
                m_scalar_quark3=gut_input[13],
                m_scalar_up=gut_input[14],
                m_scalar_top=gut_input[15],
                m_scalar_down=gut_input[16],
                m_scalar_bottom=gut_input[17],
                higgs_tanbeta=gut_input[18],
            )
        with open(f".tmp/input_{event_id}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(tmp_input)
        # build the susy command and run it
        cmd = self.softsusy_cmd + f" .tmp/input_{event_id}.csv > .tmp/susy_output_{event_id}"
        return os.system(cmd)

    def run_micromegas(
        self,
        event_id:   int,
    ):
        """
        This function runs the micromegas for a particular
        output slha file from softsusy.
        """
        cmd = self.micromegas_cmd + f" .tmp/susy_output_{event_id} .tmp/ _{event_id}"
        return os.system(cmd)

    def run_end_of_file(
        self,
        file_name: str
    ):
        """
        Plugins to be run on the entire file after all
        events have been evaluated within the file.

        One required task is for the master node to gather up all
        reco object information and append it to the arrays in the
        mssm file.

        Args:
            file_name (_type_): _description_
        """
        if self.rank == 0:
            data = np.load(self.mssm_folder + file_name)
            data_dict = {key: data[key] for key in data}
            # Extract arrays
            gut_test = data['gut_test']

            # Determine the number of entries
            num_entries = len(gut_test)
            gut_test_susy_physical_params = np.zeros((num_entries, len(softsusy_physical_parameters.keys())), dtype=np.float64)
            gut_test_susy_weak_params = np.zeros((num_entries, len(softsusy_weak_parameters.keys())), dtype=np.float64)
            gut_test_micromegas_params = np.zeros((num_entries, 66), dtype='S16')
            gut_test_gut_params = np.zeros((num_entries, 4), dtype=np.float64)
            gut_test_constraints = np.zeros((num_entries, 3), dtype=np.float64)
            gut_test_output_susy_physical_params = np.zeros(
                (num_entries, len(softsusy_physical_parameters.keys())),
                dtype=np.float64
            )
            gut_test_output_susy_weak_params = np.zeros((num_entries, len(softsusy_weak_parameters.keys())), dtype=np.float64)
            gut_test_output_micromegas_params = np.zeros((num_entries, 66), dtype='S16')
            gut_test_output_gut_params = np.zeros((num_entries, 4), dtype=np.float64)
            gut_test_output_constraints = np.zeros((num_entries, 3), dtype=np.float64)

            for i in range(1, self.size):
                distributed_data = self.comm.recv(source=i, tag=11)
                indices = distributed_data['indices']
                gut_test_susy_physical_params[indices] = distributed_data['gut_test_susy_physical_params']
                gut_test_susy_weak_params[indices] = distributed_data['gut_test_susy_weak_params']
                gut_test_micromegas_params[indices] = distributed_data['gut_test_micromegas_params']
                gut_test_gut_params[indices] = distributed_data['gut_test_gut_params']
                gut_test_constraints[indices] = distributed_data['gut_test_constraints']
                gut_test_output_susy_physical_params[indices] = distributed_data['gut_test_output_susy_physical_params']
                gut_test_output_susy_weak_params[indices] = distributed_data['gut_test_output_susy_weak_params']
                gut_test_output_micromegas_params[indices] = distributed_data['gut_test_output_micromegas_params']
                gut_test_output_gut_params[indices] = distributed_data['gut_test_output_gut_params']
                gut_test_output_constraints[indices] = distributed_data['gut_test_output_constraints']

            """Save new arrays back to file"""
            data_dict['gut_test_susy_physical_params'] = gut_test_susy_physical_params
            data_dict['gut_test_susy_weak_params'] = gut_test_susy_weak_params
            data_dict['gut_test_micromegas_params'] = gut_test_micromegas_params
            data_dict['gut_test_gut_params'] = gut_test_gut_params
            data_dict['gut_test_constraints'] = gut_test_constraints
            data_dict['gut_test_output_susy_physical_params'] = gut_test_output_susy_physical_params
            data_dict['gut_test_output_susy_weak_params'] = gut_test_output_susy_weak_params
            data_dict['gut_test_output_micromegas_params'] = gut_test_output_micromegas_params
            data_dict['gut_test_output_gut_params'] = gut_test_output_gut_params
            data_dict['gut_test_output_constraints'] = gut_test_output_constraints

            np.savez(self.mssm_folder + 'evaluated_' + file_name, **data_dict)
        else:
            self.comm.send({
                'gut_test_output_susy_physical_params': self.gut_test_output_susy_physical_params,
                'gut_test_output_susy_weak_params': self.gut_test_output_susy_weak_params,
                'gut_test_output_micromegas_params': self.gut_test_output_micromegas_params,
                'gut_test_output_gut_params': self.gut_test_output_gut_params,
                'gut_test_output_constraints': self.gut_test_output_constraints,
                'gut_test_susy_physical_params': self.gut_test_susy_physical_params,
                'gut_test_susy_weak_params': self.gut_test_susy_weak_params,
                'gut_test_micromegas_params': self.gut_test_micromegas_params,
                'gut_test_gut_params': self.gut_test_gut_params,
                'gut_test_constraints': self.gut_test_constraints,
                'indices': self.distributed_events['indices']
            }, dest=0, tag=11)

    def run_end_of_mssm(self):
        """
        Set of functions to be run at the end of the entire
        mssm job.  Some default operations are to create
        profiling plots.
        """
        if self.rank == 0:
            try:
                self.logger.info("mssm program ran successfully. Closing out.")
            except Exception as exception:
                self.report_error(exception=exception)
        else:
            pass

    def run_mssm(self):
        """
        Main mssm program loop.

        Every function call here should be encapsulated in a try/except
        block to prevent hangups with comm.Barrier.

        I. Set up input files for run
        ~ Loop over each file
            a. Load file to determine unique events and indices for each array type
            b. Run whole file over file begin plugins
            c. Set up output arrays in output h5 file
            d. Send index, output information to each worker
            ~ Loop over each event in each worker
                i. Grab event related information and pass to plugins in order
                ii. Collect output information and add to output files
            e. Run whole file over file end plugins
            f. Run end of file functions
        II. Run end of program functions
        """
        self.barrier()

        """Set up input files"""
        try:
            self.set_up_input_files()
        except Exception as exception:
            self.report_error(exception=exception)
        self.barrier()

        """Set up progress bar"""
        if self.rank == 0:
            self.logger.info(f'running mssm with {self.size} workers.')
            self.progress_bar = tqdm(
                total=len(self.mssm_files),
                ncols=100,
                colour='MAGENTA',
                leave=True,
            )
        self.barrier()

        """Loop over files and call master/worker methods for each."""
        for ii, file_name in enumerate(self.mssm_files):
            """First run begin of file"""
            if self.rank == 0:
                try:
                    self.progress_bar.set_description_str(f'File [{ii+1}/{len(self.mssm_files)}]')
                    self.run_begin_of_file(file_name)
                except Exception as exception:
                    self.report_error(exception=exception)
            self.barrier()

            """Reset event/plugin errors"""
            self.reset_event_errors()
            self.barrier()

            """Set up output arrays in mssm file"""
            if self.rank == 0:
                try:
                    self.set_up_output_arrays(file_name)
                except Exception as exception:
                    self.report_error(exception=exception)
            self.barrier()

            """Prepare indices for workers"""
            try:
                self.distribute_tasks(file_name)
            except Exception as exception:
                self.report_error(exception=exception)
            self.barrier()

            """Now process the file"""
            try:
                self.barrier()
                if self.rank == 0:
                    """Process event in master node"""
                    try:
                        self.progress_bar.set_postfix_str(f'# Events: [{self.num_events}]')
                        self.process_events_master()
                    except Exception as exception:
                        self.report_error(exception=exception)
                    self.barrier()
                else:
                    """Process event in worker node"""
                    self.barrier()
                    try:
                        self.process_events_worker()
                    except Exception as exception:
                        self.report_error(exception=exception)
                """Ensure that changes are pushed to the mssm file"""
                self.barrier()
            except Exception as exception:
                self.report_error(exception=exception)
            self.barrier()

            """Run end of file plugins"""
            try:
                self.run_end_of_file(file_name)
            except Exception as exception:
                self.report_error(exception=exception)
            self.barrier()

            """Update progress bar"""
            if self.rank == 0:
                try:
                    self.progress_bar.update(1)
                except Exception as exception:
                    self.report_error(exception=exception)
            self.barrier()

            """Report any event errors"""
            try:
                self.report_event_errors(file_name)
            except Exception as exception:
                self.report_error(exception=exception)
            self.barrier()

        """Run end of program functions"""
        if self.rank == 0:
            try:
                self.progress_bar.close()
            except Exception as exception:
                self.report_error(exception=exception)
        self.barrier()
        self.run_end_of_mssm()
