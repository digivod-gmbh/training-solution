from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import importlib
import time
import os
import mxnet as mx

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
            # TODO: Make dataset name variable
            lastExportFile = os.path.normpath(os.path.join(self.parent.lastExportDir, 'dataset.rec'))
            self.dataset_file.setText(lastExportFile)
        dataset_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_browse_btn.clicked.connect(self.dataset_browse_btn_clicked)
        self.dataset_format_label = QtWidgets.QLabel('Dataset format: unknown')
        self.dataset_format_label.setVisible(False)

        dataset_file_group = QtWidgets.QGroupBox()
        dataset_file_group.setTitle(_('Dataset file'))
        dataset_file_group_layout = QtWidgets.QGridLayout()
        dataset_file_group.setLayout(dataset_file_group_layout)
        dataset_file_group_layout.addWidget(self.dataset_file, 0, 0)
        dataset_file_group_layout.addWidget(dataset_browse_btn, 0, 1)
        dataset_file_group_layout.addWidget(self.dataset_format_label, 1, 0, 1, 2)
        layout.addWidget(dataset_file_group)

        self.output_folder = QtWidgets.QLineEdit()
        output_browse_btn = QtWidgets.QPushButton(_('Browse'))
        output_browse_btn.clicked.connect(self.output_browse_btn_clicked)

        output_folder_group = QtWidgets.QGroupBox()
        output_folder_group.setTitle(_('Output folder'))
        output_folder_group_layout = QtWidgets.QGridLayout()
        output_folder_group.setLayout(output_folder_group_layout)
        output_folder_group_layout.addWidget(self.output_folder, 0, 0)
        output_folder_group_layout.addWidget(output_browse_btn, 0, 1)
        layout.addWidget(output_folder_group)

        args_epochs_label = QtWidgets.QLabel(_('Epochs'))
        self.args_epochs = QtWidgets.QSpinBox()
        self.args_epochs.setValue(10)
        self.args_epochs.setMinimum(1)
        self.args_epochs.setMaximum(100)

        args_batch_size_label = QtWidgets.QLabel(_('Batch size'))
        self.args_batch_size = QtWidgets.QComboBox()
        self.args_batch_size.addItems(['4', '8', '16', '32', '64'])
        self.args_batch_size.setCurrentIndex(1)

        args_gpus_label = QtWidgets.QLabel(_('GPUs'))
        self.gpus = mx.test_utils.list_gpus()
        self.gpu_checkboxes = []
        for i in self.gpus:
            checkbox = QtWidgets.QCheckBox('GPU {}'.format(i))
            checkbox.setChecked(i == 0)
            self.gpu_checkboxes.append(checkbox)

        settings_group = QtWidgets.QGroupBox()
        settings_group.setTitle(_('Settings'))
        settings_group_layout = QtWidgets.QGridLayout()
        settings_group.setLayout(settings_group_layout)
        settings_group_layout.addWidget(args_epochs_label, 0, 0)
        settings_group_layout.addWidget(self.args_epochs, 0, 1)
        settings_group_layout.addWidget(args_batch_size_label, 1, 0)
        settings_group_layout.addWidget(self.args_batch_size, 1, 1)
        settings_group_layout.addWidget(args_gpus_label, 2, 0)
        row = 2
        for i, checkbox in enumerate(self.gpu_checkboxes):
            settings_group_layout.addWidget(checkbox, row, 1)
            row += 1
        layout.addWidget(settings_group)

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

    def output_browse_btn_clicked(self):
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'))
        self.output_folder.setText(output_folder)

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

        if dataset_file:
            self.dataset_format_label.setText(_('Dataset format: {}').format(filter2format[selected_filter]))
            self.dataset_format_label.setVisible(True)
            self.dataset_file.setText(dataset_file)

    def create_extension_filters(self):
        for key, val in Export.config('extensions'):
            logger.debug(key)
            logger.debug(val)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    def _yolov3(self):
        from labelme.networks import yolov3

        output_dir = self.output_folder.text()
        data_shape = 416
        batch_size = int(self.args_batch_size.currentText())
        gpus = '0'
        epochs = int(self.args_epochs.value())
        learning_rate = 0.0001
        no_random_shape = True
        dataset = self.dataset_file.text()
        dataset_dir = os.path.normpath(os.path.dirname(dataset))
        classes_list = os.path.join(dataset_dir, '{}.labels'.format(Export.config('default_dataset_name')))

        self.progress.setMaximum(epochs)

        yolov3.train_yolov3(output_dir, self.progress, data_shape=data_shape, batch_size=batch_size, gpus=gpus, 
            epochs=epochs, lr=learning_rate, no_random_shape=no_random_shape, classes_list=classes_list)

