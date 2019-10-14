import logging

import termcolor

from . import __appname__

import os
import datetime


COLORS = {
    'WARNING': 'yellow',
    'INFO': 'white',
    'DEBUG': 'blue',
    'CRITICAL': 'red',
    'ERROR': 'red',
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
        logging.Logger.__init__(self, name, logging.INFO)

        # Write log to file
        log_file_path = DEFAULT_LOG_PATH
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_format = '%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)d)'
        file_formatter = ColoredFormatter(file_format, False)
        file_log = logging.FileHandler(log_file_path)
        file_log.setFormatter(file_formatter)
        self.addHandler(file_log)

        return

    def addStreamHandler(self):
        color_formatter = ColoredFormatter(ColoredLogger.FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        self.addHandler(console)


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__appname__)
