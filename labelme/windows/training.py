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
from labelme.utils import deltree, WorkerDialog
from labelme.utils.map import Map
from labelme.extensions.thread import WorkerExecutor
from labelme.extensions.formats import *
from labelme.config import MessageType
from labelme.config import Training
from labelme.config.export import Export


class TrainingWindow(WorkerDialog):

    def __init__(self, parent=None):
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

        self.formats = QtWidgets.QComboBox()
        for key, val in Export.config('formats').items():
            self.formats.addItem(val)
        self.formats.setCurrentIndex(0)
        self.formats.currentTextChanged.connect(self.on_format_change)
        self.selected_format = list(Export.config('formats').keys())[0]

        format_group = QtWidgets.QGroupBox()
        format_group.setTitle(_('Format'))
        format_group_layout = QtWidgets.QVBoxLayout()
        format_group.setLayout(format_group_layout)
        format_group_layout.addWidget(self.formats)
        layout.addWidget(format_group)

        train_dataset_label = QtWidgets.QLabel(_('Training dataset'))
        self.train_dataset_folder = QtWidgets.QLineEdit()
        train_dataset_folder_browse_btn = QtWidgets.QPushButton(_('Browse'))
        train_dataset_folder_browse_btn.clicked.connect(self.train_dataset_folder_browse_btn_clicked)

        val_label_text = '{} ({})'.format(_('Validation dataset'), _('optional'))
        val_dataset_label = QtWidgets.QLabel(val_label_text)
        self.val_dataset_folder = QtWidgets.QLineEdit()
        val_dataset_folder_browse_btn = QtWidgets.QPushButton(_('Browse'))
        val_dataset_folder_browse_btn.clicked.connect(self.val_dataset_folder_browse_btn_clicked)

        dataset_folder_group = QtWidgets.QGroupBox()
        dataset_folder_group.setTitle(_('Datasets'))
        dataset_folder_group_layout = QtWidgets.QGridLayout()
        dataset_folder_group.setLayout(dataset_folder_group_layout)
        dataset_folder_group_layout.addWidget(train_dataset_label, 0, 0, 1, 2)
        dataset_folder_group_layout.addWidget(self.train_dataset_folder, 1, 0)
        dataset_folder_group_layout.addWidget(train_dataset_folder_browse_btn, 1, 1)
        dataset_folder_group_layout.addWidget(val_dataset_label, 2, 0, 1, 2)
        dataset_folder_group_layout.addWidget(self.val_dataset_folder, 3, 0)
        dataset_folder_group_layout.addWidget(val_dataset_folder_browse_btn, 3, 1)
        layout.addWidget(dataset_folder_group)

        self.output_folder = QtWidgets.QLineEdit()
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        self.output_folder.setText(os.path.join(project_folder, self.parent._config['project_training_folder']))
        self.output_folder.setReadOnly(True)
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
        no_gpus_available_label = QtWidgets.QLabel(_('No GPUs available'))
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
        if len(self.gpu_checkboxes) > 0:
            row = 3
            for i, checkbox in enumerate(self.gpu_checkboxes):
                settings_group_layout.addWidget(checkbox, row, 1)
                row += 1
        else:
            settings_group_layout.addWidget(no_gpus_available_label, 3, 1)
        layout.addWidget(settings_group)

        button_box = QtWidgets.QDialogButtonBox()
        training_btn = button_box.addButton(_('Start Training'), QtWidgets.QDialogButtonBox.AcceptRole)
        training_btn.clicked.connect(self.training_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def on_format_change(self, value):
        formats = Export.config('formats')
        inv_formats = Export.invertDict(formats)
        if value in inv_formats:
            self.selected_format = inv_formats[value]
            logger.debug('Selected dataset format: {}'.format(self.selected_format))
        else:
            logger.debug('Dataset format not found: {}'.format(value))

    def cancel_btn_clicked(self):
        self.close()

    def train_dataset_folder_browse_btn_clicked(self):
        dataset_folder_or_file = self.dataset_folder_browse_btn_clicked('train')
        if dataset_folder_or_file:
            self.train_dataset_folder.setText(dataset_folder_or_file)

    def val_dataset_folder_browse_btn_clicked(self):
        dataset_folder_or_file = self.dataset_folder_browse_btn_clicked('val')
        if dataset_folder_or_file:
            self.val_dataset_folder.setText(dataset_folder_or_file)

    def dataset_folder_browse_btn_clicked(self, mode='train'):
        ext_filter = False
        extension = Export.config('extensions')[self.selected_format]
        format_name = Export.config('formats')[self.selected_format]
        if extension != False:
            ext_filter = '{} {}({})'.format(format_name, _('files'), extension)
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        dataset_folder = os.path.join(project_folder, self.parent._config['project_dataset_folder'])
        if ext_filter:
            dataset_folder_or_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select dataset file'), dataset_folder, ext_filter)
        else:
            dataset_folder_or_file = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select dataset folder'), dataset_folder)
        if dataset_folder_or_file:
            dataset_folder_or_file = os.path.normpath(dataset_folder_or_file)
            key = Export.detectDatasetFormat(dataset_folder_or_file)
            logger.debug('Detected dataset format {} for directory {}'.format(key, dataset_folder_or_file))
            if key is None:
                mb = QtWidgets.QMessageBox()
                mb.warning(self, _('Training'), _('Could not detect format of selected dataset'))
                return False
            return dataset_folder_or_file
        return False

    def output_browse_btn_clicked(self):
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        training_folder = os.path.join(project_folder, self.parent._config['project_training_folder'])
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), training_folder)
        if output_folder:
            output_folder = os.path.normpath(output_folder)
            self.output_folder.setText(output_folder)

    def training_btn_clicked(self):
        # Data
        data = {
            'train_dataset': self.train_dataset_folder.text(),
            'val_dataset': self.val_dataset_folder.text(),
            'output_folder': self.output_folder.text(),
            'selected_format': self.selected_format,
            'training_name': self.training_name.text(),
            'network': self.networks.currentText(),
            'gpu_checkboxes': self.gpu_checkboxes,
            'args_epochs': self.args_epochs.value(),
            'args_batch_size': self.args_batch_size.value(),
            'args_learning_rate': self.args_learning_rate.value(),
        }

        # Execution
        executor = TrainingExecutor(data)
        self.run_thread(executor, self.finish_training)

    def finish_training(self):
        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Training'), _('Network has been trained successfully'))
        self.close()


class TrainingExecutor(WorkerExecutor):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        logger.debug('Prepare training')

        try:
            import ptvsd
            ptvsd.debug_this_thread()
        except:
            pass

        train_dataset = self.data['train_dataset']
        is_train_dataset_valid = True
        if not train_dataset:
            is_train_dataset_valid = False
        train_dataset = os.path.normpath(train_dataset)
        if not (os.path.isdir(train_dataset) or os.path.isfile(train_dataset)):
            is_train_dataset_valid = False
        if not is_train_dataset_valid:
            self.thread.message.emit(_('Training'), _('Please select a valid training dataset'), MessageType.Warning)
            self.abort()
            return

        val_dataset = self.data['val_dataset']
        is_val_dataset_valid = True
        if not val_dataset:
            is_val_dataset_valid = False
        val_dataset = os.path.normpath(val_dataset)
        if not (os.path.isdir(val_dataset) or os.path.isfile(val_dataset)):
            is_val_dataset_valid = False
        if not is_val_dataset_valid:
            # Validation dataset is optional
            val_dataset = False

        output_folder = os.path.normpath(self.data['output_folder'])
        training_name = self.data['training_name']
        training_name = re.sub(r'[^a-zA-Z0-9 _-]+', '', training_name)

        if not training_name:
            self.thread.message.emit(_('Training'), _('Please enter a valid training name'), MessageType.Warning)
            self.abort()
            return
        
        output_folder = os.path.join(output_folder, training_name)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        elif len(os.listdir(output_folder)) > 0:
            msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(output_folder)
            if self.doConfirm(_('Training'), msg, MessageType.Warning):
                deltree(output_folder)
                time.sleep(0.5) # wait for deletion to be finished
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                self.abort()
                return

        if not os.path.isdir(output_folder):
            self.thread.message.emit(_('Training'), _('The selected output directory "{}" could not be created').format(output_folder), MessageType.Warning)
            self.abort()
            return

        network = self.data['network']

        networks = Training.config('networks')
        func_name = None
        for key in networks:
            if network in networks[key]:
                func_name = key
        
        if func_name is None:
            self.thread.message.emit(_('Training'), _('Network {} could not be found').format(network), MessageType.Error)
            self.abort()
            return

        # Training settings
        gpus = []
        gpu_checkboxes = self.data['gpu_checkboxes']
        for i, gpu in enumerate(gpu_checkboxes):
            if gpu.checkState() == Qt.Checked:
                gpus.append(str(i))
        gpus = ','.join(gpus)
        epochs = int(self.data['args_epochs'])
        batch_size = int(self.data['args_batch_size'])

        # Dataset
        dataset_format = self.data['selected_format']
        train_dataset_obj = Export.config('objects')[dataset_format]()
        train_dataset_obj.setInputFolderOrFile(train_dataset)
        if val_dataset:
            val_dataset_obj = Export.config('objects')[dataset_format]()
            val_dataset_obj.setInputFolderOrFile(val_dataset)

        labels = train_dataset_obj.getLabels()
        num_train_samples = train_dataset_obj.getNumSamples()
        num_batches = int(math.ceil(num_train_samples / batch_size))
        
        args = Map({
            'train_dataset': train_dataset,
            'validate_dataset': val_dataset,
            'training_name': training_name,
            'batch_size': batch_size,
            'learning_rate': float(self.data['args_learning_rate']),
            'gpus': gpus,
            'epochs': epochs,
        })

        self.thread.update.emit(_('Loading data ...'), 0, epochs * num_batches + 5)

        network = Training.config('objects')[func_name]()
        network.setAbortable(self.abortable)
        network.setThread(self.thread)
        network.setArgs(args)
        network.setOutputFolder(output_folder)
        network.setTrainDataset(train_dataset_obj)
        network.setLabels(labels)

        if val_dataset:
            network.setValDataset(val_dataset_obj)

        self.checkAborted()

        network.training()
