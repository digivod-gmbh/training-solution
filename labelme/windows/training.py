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
from labelme.utils import Worker, TrainingObject
from labelme.utils import Application

class Training():

    @staticmethod
    def config(key = None):
        config = {
            'default_training_name': 'training',
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

        super().__init__(parent)
        self.setWindowTitle(_('Training'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        self.dataset_format_init = False

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

        dataset_train_file_label = QtWidgets.QLabel(_('Training dataset'))
        self.dataset_train_file = QtWidgets.QLineEdit()
        dataset_train_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_train_browse_btn.clicked.connect(self.dataset_train_browse_btn_clicked)

        dataset_val_file_label = QtWidgets.QLabel(_('Validation dataset'))
        self.dataset_val_file = QtWidgets.QLineEdit()
        if self.parent.exportState.lastFileVal is not None:
            self.dataset_val_file.setText(self.parent.exportState.lastFileVal)
        dataset_val_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_val_browse_btn.clicked.connect(self.dataset_val_browse_btn_clicked)
        dataset_format_label = QtWidgets.QLabel(_('Dataset format:'))
        self.dataset_format_value_label = QtWidgets.QLabel('-')

        dataset_files_group = QtWidgets.QGroupBox()
        dataset_files_group.setTitle(_('Dataset files'))
        dataset_files_group_layout = QtWidgets.QGridLayout()
        dataset_files_group.setLayout(dataset_files_group_layout)
        dataset_files_group_layout.addWidget(dataset_train_file_label, 0, 0, 1, 3)
        dataset_files_group_layout.addWidget(self.dataset_train_file, 1, 0, 1, 2)
        dataset_files_group_layout.addWidget(dataset_train_browse_btn, 1, 2)
        dataset_files_group_layout.addWidget(dataset_val_file_label, 2, 0, 1, 3)
        dataset_files_group_layout.addWidget(self.dataset_val_file, 3, 0, 1, 2)
        dataset_files_group_layout.addWidget(dataset_val_browse_btn, 3, 2)
        dataset_files_group_layout.addWidget(dataset_format_label, 4, 0)
        dataset_files_group_layout.addWidget(self.dataset_format_value_label, 4, 1, 1, 2)
        layout.addWidget(dataset_files_group)

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
        self.args_batch_size.addItems(['2', '4', '8', '16', '32', '64', '128'])
        self.args_batch_size.setCurrentIndex(2)

        args_gpus_label = QtWidgets.QLabel(_('GPUs'))
        self.gpus = mx.test_utils.list_gpus() # ['0']
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

        # Init data
        if self.parent.exportState.lastFileTrain is not None:
            self.init_dataset_file_input(True, self.parent.exportState.lastFileTrain, Export.extension2format(self.parent.exportState.lastExtension))

    def training_btn_clicked(self):
        dataset_train_file = self.dataset_train_file.text()
        if not dataset_train_file or not os.path.isfile(dataset_train_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Please select a valid training dataset file'))
            return

        network = self.networks.currentText()

        self.progress = QtWidgets.QProgressDialog(_('Training {} ...').format(network), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.NonModal)
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

        # if self.progress.wasCanceled():
        #     self.progress.close()
        #     return

        # self.progress.close()

        # mb = QtWidgets.QMessageBox
        # mb.information(self, _('Training'), _('Network {} has been trained successfully').format(network))
        # self.close()

    def cancel_btn_clicked(self):
        self.close()

    def output_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('training/last_output_dir', '')
        logger.debug('Restored value "{}" for setting training/last_output_dir'.format(last_dir))
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), last_dir)
        if output_folder:
            output_folder = os.path.normpath(output_folder)
            self.parent.settings.setValue('training/last_output_dir', output_folder)
        self.output_folder.setText(output_folder)

    def browse_dataset_file(self):
        dataset_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select dataset file'), self.parent.exportState.lastDir, Export.filters())
        if dataset_file:
            dataset_file = os.path.normpath(dataset_file)
        return dataset_file, Export.filter2format(selected_filter)

    def init_dataset_file_input(self, isTrain=True, dataset_file=None, selected_format=None):
        if dataset_file:
            if selected_format:
                self.dataset_format_value_label.setText(selected_format)
                self.dataset_format_init = True
            if isTrain:
                self.dataset_train_file.setText(dataset_file)
            if not isTrain:
                self.dataset_val_file.setText(dataset_file)

    def dataset_train_browse_btn_clicked(self):
        dataset_file, selected_format = self.browse_dataset_file()
        if selected_format != self.dataset_format_value_label.text() and self.dataset_format_init:
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Training an validation dataset must have the same format'))
        self.init_dataset_file_input(True, dataset_file, selected_format)

    def dataset_val_browse_btn_clicked(self):
        dataset_file, selected_format = self.browse_dataset_file()
        if selected_format != self.dataset_format_value_label.text() and self.dataset_format_init:
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Training an validation dataset must have the same format'))
        self.init_dataset_file_input(False, dataset_file, selected_format)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    def _yolov3(self):

        epochs = int(self.args_epochs.value())
        self.progress.setMaximum(epochs + 4)
        self.progress.setLabelText(_('Initializing training thread ...'))
        self.progress.setValue(0)

        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = TrainingObject(worker, self.start_training_progress, self.update_training_progress)
        self.progress.canceled.disconnect()
        self.progress.canceled.connect(self.abort_training_progress)
        worker.addObject(self.worker_object)
        worker.start()

    def start_training_progress(self):
        from labelme.networks import NetworkYoloV3

        output_dir = self.output_folder.text()
        batch_size = int(self.args_batch_size.currentText())
        gpus = '0' # TODO: Make configurable
        epochs = int(self.args_epochs.value())

        dataset_train_file = os.path.normpath(self.dataset_train_file.text())
        dataset_val_file = ''
        if self.dataset_val_file.text():
            dataset_val_file = os.path.normpath(self.dataset_val_file.text())

        dataset_dir = os.path.normpath(os.path.dirname(dataset_train_file))
        classes_list = os.path.join(dataset_dir, '{}.labels'.format(Export.config('default_dataset_name')))

        network = NetworkYoloV3(self.worker_object, output_dir, train_dataset=dataset_train_file, validate_dataset=dataset_val_file, 
            batch_size=batch_size, gpus=gpus, epochs=epochs, classes_list=classes_list)
        self.worker_object.setAbortFunc(network.abort)
        network.start()

    def update_training_progress(self, msg=None, value=None):
        if self.progress.wasCanceled():
            return
        if msg is not None:
            self.progress.setLabelText(msg)
        if value is not None:
            self.progress.setValue(value)

    def abort_training_progress(self):
        self.progress.setLabelText(_('Cancelling ...'))
        #self.progress.setValue(0)
        self.progress.setMaximum(0)
        self.worker_object.abort()
        worker = Application.getWorker(self.worker_idx)
        worker.wait()
        self.progress.cancel()
        Application.destroyWorker(self.worker_idx)

