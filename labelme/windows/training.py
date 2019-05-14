from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import importlib
import time
import os

from labelme.logger import logger
from labelme.windows import Export

class Training():

    @staticmethod
    def config(key = None):
        config = {
            'networks': {
                '_yolov3': _('YoloV3')
            }
        }
        if key is not None:
            if key in config:
                return config[key]
        return config


class TrainingWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super(TrainingWindow, self).__init__(parent)
        self.setWindowTitle(_('Training'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.networks = QtWidgets.QComboBox()
        for key, val in Training.config('networks').items():
            self.networks.addItem(val)

        network_group = QtWidgets.QGroupBox()
        network_group.setTitle(_('Network'))
        network_group_layout = QtWidgets.QVBoxLayout()
        network_group.setLayout(network_group_layout)
        network_group_layout.addWidget(self.networks)
        layout.addWidget(network_group)

        self.dataset_file = QtWidgets.QLineEdit()
        if self.parent.lastExportDir is not None:
            self.dataset_file.setText(self.parent.lastExportDir)
        dataset_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_browse_btn.clicked.connect(self.dataset_browse_btn_clicked)
        self.dataset_format_label = QtWidgets.QLabel('')

        dataset_file_group = QtWidgets.QGroupBox()
        dataset_file_group.setTitle(_('Dataset file'))
        dataset_file_group_layout = QtWidgets.QGridLayout()
        dataset_file_group.setLayout(dataset_file_group_layout)
        dataset_file_group_layout.addWidget(self.dataset_file, 0, 0)
        dataset_file_group_layout.addWidget(dataset_browse_btn, 0, 1)
        dataset_file_group_layout.addWidget(self.dataset_format_label, 1, 0, 1, 2)
        layout.addWidget(dataset_file_group)

        button_box = QtWidgets.QDialogButtonBox()
        training_btn = button_box.addButton(_('Start Training'), QtWidgets.QDialogButtonBox.AcceptRole)
        training_btn.clicked.connect(self.training_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def training_btn_clicked(self):
        dataset_file = self.dataset_file.text()
        if not dataset_file or not os.path.isfile(dataset_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Please select a valid dataset file'))
            return

        network = self.networks.currentText()

        self.progress = QtWidgets.QProgressDialog(_('Training {} ...').format(network), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setValue(0)
        self.progress.show()

        networks = Training.config('networks')
        func_name = None
        for key in networks:
            if network in networks[key]:
                func_name = key
        
        if func_name is None:
            logger.error('Network {} could not be found'.format(val))
            return

        key = self.networks.currentText()
        if key in Training.config('networks'):
            func_name = Training.config('networks')[key]
            training_func = getattr(self, func_name)
            training_func()

        if self.progress.wasCanceled():
            self.progress.close()
            return

        self.progress.close()

        mb = QtWidgets.QMessageBox
        mb.information(self, _('Training'), _('Network {} has been trained successfully').format(network))
        self.close()

    def cancel_btn_clicked(self):
        self.close()

    def dataset_browse_btn_clicked(self):
        formats = Export.config('formats')
        extensions = Export.config('extensions')
        filters = []
        filter2format = {}
        for key in formats:
            f = '{} (*{})'.format(formats[key], extensions[key])
            filters.append(f)
            filter2format[f] = formats[key]
        filters = ';;'.join(filters)

        dataset_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select dataset file'), self.parent.lastExportDir, filters)

        self.dataset_format_label.setText(_('Dataset format: {}').format(filter2format[selected_filter]))
        self.dataset_file.setText(dataset_file)

    def create_extension_filters(self):
        for key, val in Export.config('extensions'):
            logger.debug(key)
            logger.debug(val)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    def _yolov3(self):
        time.sleep(1)