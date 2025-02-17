"""
corenet main program
"""
import argparse

from corenet.programs.wrapper import parse_command_line_config


def run():
    """
    corenet main program.
    """
    parser = argparse.ArgumentParser(
        prog='CORENet Module Runner',
        description='This program constructs a corenet module ' +
                    'from a config file, and then runs the set of modules ' +
                    'in the configuration.',
        epilog='...'
    )
    parser.add_argument(
        'config_file', metavar='<str>.yml', type=str,
        help='config file specification for a corenet module.'
    )
    parser.add_argument(
        '-n', dest='name', default=None,
        help='name for this run (default None).'
    )
    parser.add_argument(
        '-scratch', dest='local_scratch', default='/local_scratch',
        help='location for the local scratch directory.'
    )
    parser.add_argument(
        '-corenet', dest='local_corenet', default='/local_corenet',
        help='location for the local corenet directory.'
    )
    parser.add_argument(
        '-data', dest='local_data', default='/local_data',
        help='location for the local data directory.'
    )
    parser.add_argument(
        '-anomaly', dest='anomaly', default=False,
        help='enable anomaly detection in pytorch'
    )
    parser.add_argument(
        '-dataset_folder', dest='dataset_folder', default='',
        help='custom dataset folder location'
    )
    args = parser.parse_args()
    meta, module_handler = parse_command_line_config(args)
    module_handler.run_modules()


if __name__ == "__main__":
    run()
