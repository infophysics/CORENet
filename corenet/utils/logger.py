"""
Logger class for all blip classes.
"""
import warnings
import logging
import platform
import traceback
import socket
import re
import uuid
import psutil
import os


class CORENetError(Exception):
    """Custom default error for corenet"""
    def __init__(
        self,
        message="An error occurred"
    ):
        self.message = message
        super().__init__(self.message)


class EventError(Exception):
    """Custom default error for an event"""
    def __init__(
        self,
        message="An error occurred"
    ):
        self.message = message
        super().__init__(self.message)


logging_level = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

warning_list = {
    "deprecation": DeprecationWarning,
    "import": ImportWarning,
    "resource": ResourceWarning,
    "user": UserWarning,
}

error_list = {
    "attribute": AttributeError,
    "corenet": CORENetError,
    "event": EventError,
    "index": IndexError,
    "file": FileExistsError,
    "memory": MemoryError,
    "runtime": RuntimeError,
    "type": TypeError,
    "value": ValueError,
}


class LoggingFormatter(logging.Formatter):
    """
    Formatting for corenet logging.

    Args:
        logging (_logging.Formatter_): The formatter
        to be edited.

    Returns:
        _logging.Formatter_: The edited formatter
    """
    console_extra = " [%(name)s]: %(message)s"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[1;34m"
    purple = "\x1b[1;35m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    FORMATS = {
        logging.DEBUG: "[" + grey + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.INFO: "[" + blue + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.WARNING: "[" + yellow + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.ERROR: "[" + red + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
        logging.CRITICAL: "[" + bold_red + "%(levelname)s" + reset + "] [" + purple + "%(name)s" + reset + "]: %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    """
    Custom logging wrapper for corenet programs.
    The logger writes to three different streams:
        (1) console
        (2) file
        (3) debug
    """

    def __init__(
        self,
        name: str = "default",
        level: str = "debug",
    ):
        """
        Initializer for the logger

        Args:
            name (str, optional): name for this logger (will appear
            after the log message type [ERROR] [name]). Defaults to "default".

            level (str, optional): _description_. Defaults to "debug".
            output (str, optional): whether to log to console, file
            or both. Defaults to "file".

            output_file (str, optional): name of the output file. Defaults to "log".
            file_mode (str, optional): whether to append, or rewrite
            log files for this run. Defaults to "a".

        Raises:
            ValueError: _description_
        """
        # check for mistakes
        if level not in logging_level.keys():
            raise ValueError(f"Logging level {level} not in {logging_level}.")

        # create the logging directory
        if "LOCAL_SCRATCH" in os.environ.keys():
            self.local_log_dir = os.environ["LOCAL_SCRATCH"] + "/.logs/"
        else:
            self.local_log_dir = "/local_scratch/.logs/"
        if not os.path.isdir(self.local_log_dir):
            os.makedirs(self.local_log_dir)

        # use the name as the default output file name
        self.name = name
        self.level = logging_level[level]
        self.output_file = "corenet"
        self.file_mode = "a"

        # create logger
        self.logger = logging.getLogger(self.name)
        self.debug_logger = logging.getLogger(self.name + "_debug")

        # set level
        self.logger.setLevel(self.level)
        self.debug_logger.setLevel(self.level)

        # set format
        self.dateformat = "%H:%M:%S"

        self.console_formatter = LoggingFormatter()
        self.file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", self.dateformat
        )

        self.debug = logging.FileHandler(
            self.local_log_dir + "/" + self.output_file + ".debug", mode="a"
        )
        self.debug.setLevel(self.level)

        # create handler
        self.console = logging.StreamHandler()
        self.console.setLevel(self.level)
        self.console.setFormatter(self.console_formatter)
        self.logger.addHandler(self.console)
        self.file = logging.FileHandler(
            self.local_log_dir + "/" + self.output_file + ".log", mode="a"
        )
        self.file.setLevel(logging.DEBUG)
        self.file.setFormatter(self.file_formatter)
        self.logger.addHandler(self.file)
        self.debug.setFormatter(self.file)
        self.debug_logger.addHandler(self.debug)
        self.logger.propagate = False

    def info(
        self,
        message: str,
    ):
        """_summary_

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "info" """
        return self.logger.info(message)

    def debug(
        self,
        message: str,
    ):
        """_summary_

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "debug" """
        return self.debug_logger.debug(message)

    def warn(
        self,
        message: str,
    ):
        """_summary_

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "warning" """
        return self.logger.warning(message)

    def warning(
        self,
        message: str,
        warning_type: str = "user",
    ):
        """_summary_

        Args:
            message (str): _description_
            warning_type (str, optional): _description_. Defaults to "user".

        Returns:
            _type_: _description_
        """
        """Output to the standard logger "warning" """
        formatted_lines = traceback.format_stack()[-2]
        if warning_type not in warning_list.keys():
            warning_type = "user"
        self.logger.warning(message)
        warnings.warn(
            f"traceback: {formatted_lines}\nerror: {message}",
            warning_list[warning_type],
        )
        return

    def error(
        self,
        message: str,
        error_type: str = "corenet",
    ):
        """_summary_

        Args:
            message (str): _description_
            error_type (str, optional): _description_. Defaults to "value".

        Raises:
            error_list: _description_
        """
        """Output to the standard logger "error" """
        formatted_traceback = ''.join(traceback.format_stack())
        if error_type not in error_list.keys():
            error_type = "corenet"
        log_message = f"Traceback: \n{formatted_traceback}\nError: {message}"
        self.logger.error(message)
        raise error_list[error_type](log_message)

    def critical(
        self,
        message: str
    ):
        """
        """
        return self.logger.critical(message)

    def get_system_info(
        self
    ) -> dict:
        """
        Attempt to get system info using various
        python packages.  If this fails, return
        an empty dictionary.

        Returns:
            _dict_: dictionary containing system info.
        """
        info = {}
        try:
            info["platform"] = platform.system()
            info["platform-release"] = platform.release()
            info["platform-version"] = platform.version()
            info["architecture"] = platform.machine()
            info["hostname"] = socket.gethostname()
            info["ip-address"] = socket.gethostbyname(socket.gethostname())
            info["mac-address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
            info["processor"] = platform.processor()
            info["physical_cores"] = psutil.cpu_count(logical=False)
            info["logical_cores"] = psutil.cpu_count(logical=True)
            info["local_scratch"] = str(round(psutil.disk_usage('/local_scratch').free / (1024.0**3))) + " GB"
            info["local_data"] = str(round(psutil.disk_usage('/local_data').free / (1024.0**3))) + " GB"
            info["RAM"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        except Exception as e:
            self.logger.warning(f"Unable to obtain system information: {e}.")
        return info
