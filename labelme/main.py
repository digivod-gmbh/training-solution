import argparse
import codecs
import logging
import os
import sys
import gettext
import time
import traceback
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from qtpy import QtWidgets
from qtpy import QtGui

from labelme import __appname__
from labelme import __version__
from labelme.app import MainWindow
from labelme.config import get_config
from labelme.logger import logger
from labelme.utils import newIcon

LOG_LEVEL = None

def main():
    try:
        _main()
    except Exception as e:
        logger.error(e)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', '-V', action='store_true', help='show version'
    )
    parser.add_argument(
        '--reset-config', action='store_true', help='reset qt config'
    )
    parser.add_argument(
        '--logger-level',
        default='info',
        choices=['debug', 'info', 'warning', 'fatal', 'error'],
        help='logger level',
    )
    parser.add_argument('filename', nargs='?', help='image or label filename')
    parser.add_argument(
        '--output',
        '-O',
        '-o',
        help='output file or directory (if it ends with .json it is '
             'recognized as file, else as directory)'
    )
    default_config_file = os.path.join(os.path.expanduser('~'), '.labelmerc')
    parser.add_argument(
        '--config',
        dest='config_file',
        help='config file (default: %s)' % default_config_file,
        default=default_config_file,
    )
    # config for the gui
    parser.add_argument(
        '--nodata',
        dest='store_data',
        action='store_false',
        help='stop storing image data to JSON file',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--autosave',
        dest='auto_save',
        action='store_true',
        help='auto save',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--nosortlabels',
        dest='sort_labels',
        action='store_false',
        help='stop sorting labels',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--flags',
        help='comma separated list of flags OR file containing flags',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--labels',
        help='comma separated list of labels OR file containing labels',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--validatelabel',
        dest='validate_label',
        choices=['exact', 'instance'],
        help='label validation types',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--keep-prev',
        action='store_true',
        help='keep annotation of previous frame',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        help='epsilon to find nearest vertex on canvas',
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.version:
        print('{0} {1}'.format(__appname__, __version__))
        sys.exit(0)

    global LOG_LEVEL
    LOG_LEVEL = args.logger_level.upper()
    logger.setLevel(getattr(logging, args.logger_level.upper()))

    if hasattr(args, 'flags'):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, 'r', encoding='utf-8') as f:
                args.flags = [l.strip() for l in f if l.strip()]
        else:
            args.flags = [l for l in args.flags.split(',') if l]

    if hasattr(args, 'labels'):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, 'r', encoding='utf-8') as f:
                args.labels = [l.strip() for l in f if l.strip()]
        else:
            args.labels = [l for l in args.labels.split(',') if l]

    config_from_args = args.__dict__
    config_from_args.pop('version')
    reset_config = config_from_args.pop('reset_config')
    filename = config_from_args.pop('filename')
    output = config_from_args.pop('output')
    config_file = config_from_args.pop('config_file')
    config = get_config(config_from_args, config_file)

    # localization
    current_path = os.path.dirname(os.path.abspath(__file__))
    locale_dir = os.path.join(current_path, 'locale')
    if os.path.isfile(os.path.join(locale_dir, config['language'], 'LC_MESSAGES', 'labelme.po')):
        lang = gettext.translation('labelme', localedir=locale_dir, languages=[config['language']])
        lang.install()
    else:
        gettext.install('labelme')

    if not config['labels'] and config['validate_label']:
        logger.error('--labels must be specified with --validatelabel or '
                     'validate_label: true in the config file '
                     '(ex. ~/.labelmerc).')
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith('.json'):
            output_file = output
        else:
            output_dir = output

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon('icon'))
    win = MainWindow(
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info('Resetting Qt config: %s' % win.settings.fileName())
        win.settings.clear()
        sys.exit(0)

    win.show()
    win.raise_()
    sys.excepthook = excepthook
    sys.exit(app.exec_())


def excepthook(exctype, excvalue, tracebackobj):

    app = QtWidgets.QApplication.instance()
    app.closeAllWindows()

    tbinfofile = StringIO()
    traceback.print_tb(tracebackobj, None, tbinfofile)
    tbinfofile.seek(0)
    tbinfo = tbinfofile.read()
    notice = _('An error occured with following message')
    errmsg = '%s:\n\n%s' % (notice, str(excvalue))
    
    global LOG_LEVEL
    if LOG_LEVEL == 'DEBUG':
        msg = '\n'.join([errmsg, '-' * 80, tbinfo])
        logger.error(msg)
    else:
        msg = errmsg
        logger.error(errmsg)

    errorbox = QtWidgets.QMessageBox()
    errorbox.setText(msg)
    errorbox.exec_()


# this main block is required to generate executable by pyinstaller
if __name__ == '__main__':
    main()
