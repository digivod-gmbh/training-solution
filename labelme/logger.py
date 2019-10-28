import logging

import termcolor

from . import __appname__

import os
import datetime


# Log levels:
# NOTSET     0
# DEBUG     10
# INFO      20
# WARNING   30
# ERROR     40
# CRITICAL  50

COLORS = {
    'DEBUG': 'blue',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'magenta',
}
DEFAULT_LOG_PATH = os.path.expanduser('~/.digivod/training_solution_{}.log'.format(datetime.date.today().strftime('%Y_%m_%d')))
DEFAULT_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class ColoredFormatter(logging.Formatter):

    date_format = DEFAULT_LOG_DATE_FORMAT

    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg, ColoredFormatter.date_format)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            colored_levelname = termcolor.colored(
                '[{}]'.format(levelname), color=COLORS[levelname]
            )
            record.levelname = colored_levelname
        return logging.Formatter.format(self, record)


class ColoredLogger(logging.Logger):

    fmt_filename = termcolor.colored('%(filename)s', attrs={'bold': True})
    FORMAT = '%(asctime)s %(levelname)s %(message)s ({}:%(lineno)d)'.format(fmt_filename)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        # Write log to file
        log_file_path = DEFAULT_LOG_PATH
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_format = '%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)d)'
        file_formatter = ColoredFormatter(file_format, False)
        file_log = logging.FileHandler(log_file_path)
        file_log.setFormatter(file_formatter)
        file_log.setLevel(logging.DEBUG)  # Always log to file
        self.addHandler(file_log)

        return

    def addStreamHandler(self, level=logging.INFO):
        color_formatter = ColoredFormatter(ColoredLogger.FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        console.setLevel(level)
        self.addHandler(console)


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__appname__)
