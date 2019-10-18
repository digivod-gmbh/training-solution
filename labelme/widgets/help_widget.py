import os
import json
import traceback

from qtpy import QtWidgets

from labelme.config import get_config
from labelme.logger import logger
from labelme.utils.qt import newIcon


class HelpWidget(QtWidgets.QWidget):

    def __init__(self, help_key, widget_func, parent=None):
        super().__init__(parent)
        self.initialize(help_key, widget_func)

    def initialize(self, help_key, widget_func):
        self.help_key = help_key

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.widget = widget_func(self.layout)

        self.help_button = self.get_help_button()
        self.layout.addWidget(self.help_button)

        self.layout.addStretch()

    def get_help_button(self, parent=None):
        help_button = QtWidgets.QPushButton()
        help_button.setParent(parent)
        help_button.setIcon(newIcon('question'))
        help_button.clicked.connect(self.on_click)
        help_button.setStyleSheet('border:0;background:transparent;padding:0;margin:0 0 6px 0')
        help_button.resize(32, 32)
        return help_button

    def on_click(self):
        help_text = HelpWidget.getText(self.help_key)
        mb = QtWidgets.QMessageBox
        mb.information(self, _('Help'), help_text)

    @staticmethod
    def getText(key):
        try:
            config = get_config()
            current_path = os.path.dirname(os.path.abspath(__file__))
            locale_dir = os.path.join(current_path, '..', 'locale')
            local_file = os.path.join(locale_dir, config['language'], 'LC_MESSAGES', 'help_messages.json')
            global_file = os.path.join(locale_dir, 'help_messages.json')
            language_file = global_file
            if os.path.isfile(local_file):
                language_file = local_file
            if not os.path.isfile(language_file):
                logger.error('Help file {} not found'.format(language_file))
                return _('Help file could not be found')
            with open(local_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if key in data.keys():
                    return data[key]
            logger.error('No content found for help with key {}'.format(key))
        except Exception as e:
            logger.error('Error while obtaining help message: {}'.format(traceback.format_exc()))
        return ''


class HelpLabel(HelpWidget):

    def __init__(self, help_key, label_text='', parent=None):
        self.label_text = label_text
        super().__init__(help_key, self.addLabel, parent)

    def addLabel(self, layout):
        widget = QtWidgets.QLabel(self.label_text)
        layout.addWidget(widget)
        return widget


class HelpCheckbox(HelpWidget):

    def __init__(self, help_key, label_text='', parent=None):
        self.label_text = label_text
        super().__init__(help_key, self.addCheckbox, parent)

    def addCheckbox(self, layout):
        widget = QtWidgets.QCheckBox(self.label_text)
        layout.addWidget(widget)
        return widget


class HelpGroupBox(HelpWidget):

    def __init__(self, help_key, label_text='', parent=None):
        self.label_text = label_text
        super().__init__(help_key, None, parent)

    def initialize(self, help_key, widget_func):
        self.help_key = help_key

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 5, 0, 0)
        self.setLayout(self.layout)

        self.help_button = self.get_help_button(self)

        self.widget = QtWidgets.QGroupBox()
        self.widget.setTitle(self.label_text)
        self.layout.addWidget(self.widget)

        tmp_label = QtWidgets.QLabel(self.label_text)
        width = tmp_label.fontMetrics().boundingRect(self.label_text).width()
        self.help_button.move(width + 5, -5)

        self.help_button.raise_()

        self.layout.addStretch()
