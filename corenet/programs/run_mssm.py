"""
mssmND main program
"""
import argparse
from mpi4py import MPI

from corenet.utils.logger import Logger
from corenet.utils.config import ConfigParser
from corenet.programs.mssm import MSSM


def run():
    """
    This program runs the mssm module from a config file.
    It utilizes MPI + H5 to distribute the processing of events
    over multiple cores which greatly speeds up runtime.  When 
    used in conjunction with the "create_mssm_runs" program,
    running a post-flow mssm can performed in minutes.
    """

    """Create a logger instance"""
    logger = Logger("mssm_runner")

    """
    We do a preliminary check to ensure that MPI is available
    and that the number of processes is at least 2.  Assuming
    that mssmND is being run on a system with more than one
    CPU core.
    """
    try:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        """Check that size >= 2, which is required for mssm to run.  Otherwise we quit early"""
        if size < 2:
            logger.error(f"number of processes must be >=2! Received {size} from MPI", "value")
    except Exception as e:
        logger.error(f"error occurred with gathering MPI: {e}", "runtime")

    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        prog="mssm Module Runner",
        description="This program runs the mssm module from a config file. \
            It utilizes MPI + H5 to distribute the processing of events \
            over multiple cores which greatly speeds up runtime.  When \
            used in conjunction with the 'create_mssm_runs' program, \
            running a post-flow mssm can performed in minutes.",
        epilog="...",
    )
    parser.add_argument(
        "config_file",
        metavar="<config_file>.yml",
        type=str,
        help="config file specification for this mssm module.",
    )
    parser.add_argument(
        "-name",
        dest="name",
        default="mssm",
        help='name for this run (default "mssm").',
    )
    parser.add_argument(
        "-n",
        dest="number_of_files",
        default=None,
        help='number of files to process (default None).',
    )

    """Parse command line arguments"""
    args = parser.parse_args()
    config_file = args.config_file
    name = args.name
    number_of_files = args.number_of_files

    """Parse the config file"""
    try:
        config = ConfigParser(config_file).data
    except Exception as e:
        logger.error(f"failed to parse config: {e}")

    """Determine whether to only pick a subset of files"""
    if number_of_files is not None:
        try:
            number_of_files = int(number_of_files)
        except Exception as e:
            logger.error(f"unable to convert number_of_files from {type(number_of_files)} to int: {e}")

        if isinstance(number_of_files, int):
            if number_of_files > 0:
                config["mssm"]["number_of_files"] = number_of_files

    """Construct meta dictionary"""
    meta = {
        "name": name,
        "config_file": config_file
    }

    """Create the mssm instance"""
    try:
        mssm = MSSM(config, meta)
    except Exception as e:
        logger.error(f"failed to construct mssm object: {e}")

    """Run mssm"""
    mssm.run_mssm()


if __name__ == "__main__":
    run()
