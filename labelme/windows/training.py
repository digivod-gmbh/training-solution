from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import importlib
import time
import os
import re
import mxnet as mx
import json
import math

from labelme.logger import logger
from labelme.utils import Worker, ProgressObject, Application
from labelme.utils.map import Map
from labelme.config import Export, Training


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

        self.dataset_folder = QtWidgets.QLineEdit()
        dataset_folder_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_folder_browse_btn.clicked.connect(self.dataset_folder_browse_btn_clicked)

        dataset_folder_group = QtWidgets.QGroupBox()
        dataset_folder_group.setTitle(_('Dataset folder'))
        dataset_folder_group_layout = QtWidgets.QGridLayout()
        dataset_folder_group.setLayout(dataset_folder_group_layout)
        dataset_folder_group_layout.addWidget(self.dataset_folder, 0, 0)
        dataset_folder_group_layout.addWidget(dataset_folder_browse_btn, 0, 1)
        layout.addWidget(dataset_folder_group)

        self.output_folder = QtWidgets.QLineEdit()
        output_browse_btn = QtWidgets.QPushButton(_('Browse'))
        output_browse_btn.clicked.connect(self.output_browse_btn_clicked)

        training_name_label = QtWidgets.QLabel(_('Training name'))
        self.training_name = QtWidgets.QLineEdit()

        output_folder_group = QtWidgets.QGroupBox()
        output_folder_group.setTitle(_('Output folder'))
        output_folder_group_layout = QtWidgets.QGridLayout()
        output_folder_group.setLayout(output_folder_group_layout)
        output_folder_group_layout.addWidget(self.output_folder, 0, 0, 1, 2)
        output_folder_group_layout.addWidget(output_browse_btn, 0, 2)
        output_folder_group_layout.addWidget(training_name_label, 1, 0, 1, 3)
        output_folder_group_layout.addWidget(self.training_name, 2, 0, 1, 3)
        layout.addWidget(output_folder_group)

        args_epochs_label = QtWidgets.QLabel(_('Epochs'))
        self.args_epochs = QtWidgets.QSpinBox()
        self.args_epochs.setValue(10)
        self.args_epochs.setMinimum(1)
        self.args_epochs.setMaximum(100)

        args_batch_size_label = QtWidgets.QLabel(_('Batch size'))
        self.args_batch_size = QtWidgets.QSpinBox()
        self.args_batch_size.setValue(8)
        self.args_batch_size.setMinimum(1)
        self.args_batch_size.setMaximum(100)

        args_learning_rate_label = QtWidgets.QLabel(_('Learning rate'))
        self.args_learning_rate = QtWidgets.QDoubleSpinBox()
        self.args_learning_rate.setMinimum(1e-7)
        self.args_learning_rate.setMaximum(1.0)
        self.args_learning_rate.setSingleStep(1e-7)
        self.args_learning_rate.setDecimals(7)
        self.args_learning_rate.setValue(0.0001)

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
        settings_group_layout.addWidget(args_learning_rate_label, 2, 0)
        settings_group_layout.addWidget(self.args_learning_rate, 2, 1)

        settings_group_layout.addWidget(args_gpus_label, 3, 0)
        row = 3
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
        dataset_folder = self.dataset_folder.text()
        if not dataset_folder or not os.path.isdir(dataset_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Please select a valid dataset file'))
            return

        dataset_format = Export.detectDatasetFormat(dataset_folder)
        logger.debug('Detected dataset format {} for directory {}'.format(dataset_format, dataset_folder))
        if dataset_format is None:
            mb = QtWidgets.QMessageBox()
            mb.warning(self, _('Training'), _('Could not detect format of selected dataset'))
            return

        output_folder = os.path.normpath(self.output_folder.text())
        training_name = self.training_name.text()
        training_name = re.sub(r'[^a-zA-Z0-9 _-]+', '', training_name)

        if not training_name:
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Please enter a valid training name'))
            return
        
        output_folder = os.path.join(output_folder, training_name)
        if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('The selected output directory "{}" is not empty. Please choose an empty directory for training').format(output_folder))
            return

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        else:
            mb = QtWidgets.QMessageBox
            msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(output_folder)
            clicked_btn = mb.warning(self, _('Training'), msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if clicked_btn != QtWidgets.QMessageBox.Yes:
                return
            else:
                import shutil
                shutil.rmtree(output_folder)
                os.makedirs(output_folder)

        if not os.path.isdir(output_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('The selected output directory "{}" could not be created').format(output_folder))
            return

        network = self.networks.currentText()

        self.progress = QtWidgets.QProgressDialog(_('Initializing ...'), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.NonModal)
        self.progress.show()

        networks = Training.config('networks')
        func_name = None
        for key in networks:
            if network in networks[key]:
                func_name = key
        
        if func_name is None:
            logger.error('Network {} could not be found'.format(val))
            return

        # Training settings
        gpus = []
        for i, gpu in enumerate(self.gpu_checkboxes):
            if gpu.checkState() == Qt.Checked:
                gpus.append(str(i))
        gpus = ','.join(gpus)
        epochs = int(self.args_epochs.value())
        batch_size = int(self.args_batch_size.value())

        # Dataset
        dataset = Export.config('objects')[dataset_format]()
        label_file = dataset.getLabelFile(dataset_folder)
        train_dataset = dataset.getTrainFile(dataset_folder)
        val_dataset = dataset.getValFile(dataset_folder)

        config_file = os.path.join(dataset_folder, Export.config('config_file'))
        dataset_config = dataset.loadConfig(config_file)
        logger.debug(dataset_config)
        num_train_samples = dataset_config.samples['train']
        num_batches = int(math.ceil(num_train_samples / batch_size))
        
        args = Map({
            'train_dataset': train_dataset,
            'validate_dataset': val_dataset,
            'training_name': training_name,
            'batch_size': batch_size,
            'learning_rate': float(self.args_learning_rate.value()),
            'gpus': gpus,
            'epochs': epochs,
        })

        self.progress.setMaximum(epochs * num_batches + 5)
        self.progress.setLabelText(_('Loading data ...'))
        self.progress.setValue(0)

        network = Training.config('objects')[func_name]()
        network.setArgs(args)
        network.setOutputFolder(output_folder)
        network.setLabelFile(label_file)

        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = ProgressObject(worker, network.training, self.error_training_progress, network.abort, 
            self.update_training_progress, self.finish_training_progress)
        network.setThread(self.worker_object)

        self.progress.canceled.disconnect()
        self.progress.canceled.connect(self.abort_training_progress)
        worker.addObject(self.worker_object)
        worker.start()

    def cancel_btn_clicked(self):
        self.close()

    def dataset_folder_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('training/last_dataset_dir', '')
        logger.debug('Restored value "{}" for setting training/last_dataset_dir'.format(last_dir))
        dataset_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select dataset folder'), last_dir)
        if dataset_folder:
            dataset_folder = os.path.normpath(dataset_folder)
            key = Export.detectDatasetFormat(dataset_folder)
            logger.debug('Detected dataset format {} for directory {}'.format(key, dataset_folder))
            if key is None:
                mb = QtWidgets.QMessageBox()
                mb.warning(self, _('Training'), _('Could not detect format of selected dataset'))
                return
            self.parent.settings.setValue('training/last_dataset_dir', dataset_folder)
            self.dataset_folder.setText(dataset_folder)

    def output_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('training/last_output_dir', '')
        logger.debug('Restored value "{}" for setting training/last_output_dir'.format(last_dir))
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), last_dir)
        if output_folder:
            output_folder = os.path.normpath(output_folder)
            self.parent.settings.setValue('training/last_output_dir', os.path.dirname(output_folder))
            self.output_folder.setText(output_folder)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    def update_training_progress(self, msg=None, value=None):
        if self.progress.wasCanceled():
            return
        if msg:
            self.progress.setLabelText(msg)
        if value is not None:
            self.progress.setValue(value)
        if value == -1:
            val = self.progress.value() + 1
            self.progress.setValue(val)

    def abort_training_progress(self):
        self.progress.setLabelText(_('Cancelling ...'))
        self.progress.setMaximum(0)
        self.worker_object.abort()
        worker = Application.getWorker(self.worker_idx)
        worker.wait()
        self.progress.cancel()
        Application.destroyWorker(self.worker_idx)

    def finish_training_progress(self):
        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Training'), _('Network has been trained successfully'))
        self.close()

    def error_training_progress(self, e):
        self.progress.cancel()
        mb = QtWidgets.QMessageBox()
        mb.warning(self, _('Training'), _('An error occured during training of network'))
