from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import importlib
import time
import os
import mxnet as mx
import json

from labelme.logger import logger
from labelme.windows import Export
from labelme.utils import Worker, TrainingObject, Application
from labelme.utils.map import Map
from labelme.networks import NetworkYoloV3


class Training():

    @staticmethod
    def config(key = None):
        config = {
            'config_file_extension': '.train',
            'networks': {
                'yolov3': _('YoloV3')
            },
            'objects': {
                'yolov3': lambda: NetworkYoloV3()
            }
        }
        if key is not None:
            if key in config:
                return config[key]
            return None
        return config

    @staticmethod
    def create_training_config(config_file, network, dataset_format, label_list, datasets, args):
        data = {
            'network': network,
            'dataset_format': dataset_format,
            'label_list': label_list,
            'datasets': datasets,
            'args': args,
            'architecture': None,
            'weights': None,
        }
        logger.debug('Create training config: {}'.format(data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved training config in file: {}'.format(config_file))
    
    @staticmethod
    def update_training_config(config_file, new_data):
        old_data = {}
        with open(config_file, 'r') as f:
            old_data = json.loads(f.read())
            logger.debug('Loaded training config: {}'.format(old_data))
        data = old_data.copy()
        data.update(new_data)
        logger.debug('Update training config: {}'.format(new_data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved training config in file: {}'.format(config_file))

    @staticmethod
    def read_training_config(config_file):
        data = {}
        with open(config_file, 'r') as f:
            data = json.loads(f.read())
            logger.debug('Read training config: {}'.format(data))
        return Map(data)


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

        self.dataset_file = QtWidgets.QLineEdit()
        dataset_file_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_file_browse_btn.clicked.connect(self.dataset_file_browse_btn_clicked)

        dataset_file_group = QtWidgets.QGroupBox()
        dataset_file_group.setTitle(_('Dataset'))
        dataset_file_group_layout = QtWidgets.QGridLayout()
        dataset_file_group.setLayout(dataset_file_group_layout)
        dataset_file_group_layout.addWidget(self.dataset_file, 0, 0)
        dataset_file_group_layout.addWidget(dataset_file_browse_btn, 0, 1)
        layout.addWidget(dataset_file_group)

        self.output_file = QtWidgets.QLineEdit()
        output_browse_btn = QtWidgets.QPushButton(_('Browse'))
        output_browse_btn.clicked.connect(self.output_browse_btn_clicked)

        output_file_group = QtWidgets.QGroupBox()
        output_file_group.setTitle(_('Output file'))
        output_file_group_layout = QtWidgets.QGridLayout()
        output_file_group.setLayout(output_file_group_layout)
        output_file_group_layout.addWidget(self.output_file, 0, 0)
        output_file_group_layout.addWidget(output_browse_btn, 0, 1)
        layout.addWidget(output_file_group)

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

    def training_btn_clicked(self):
        dataset_file = self.dataset_file.text()
        dataset_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
        dataset_dir = os.path.dirname(dataset_file)
        if not dataset_file or not os.path.isfile(dataset_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Please select a valid dataset file'))
            return

        output_file = self.output_file.text()
        output_file_name = os.path.splitext(os.path.basename(output_file))[0]
        output_dir = os.path.dirname(output_file)
        if not output_dir or not os.path.isdir(output_dir):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Please select a valid output folder'))
            return

        if len(os.listdir(output_dir)) > 0:
            mb = QtWidgets.QMessageBox
            msg = _('The selected output directory "{}" is not empty. Containing files could be overwritten. Are you sure to continue?').format(output_dir)
            clicked_btn = mb.warning(self, _('Training'), msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if clicked_btn == QtWidgets.QMessageBox.No:
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

        # Training settings
        dataset_data = Export.read_dataset_config(dataset_file)
        network_idx = func_name
        gpus = []
        for i, gpu in enumerate(self.gpu_checkboxes):
            if gpu.checkState() == Qt.Checked:
                gpus.append(str(i))
        gpus = ','.join(gpus)
        dataset_dir = os.path.normpath(os.path.dirname(dataset_file))
        label_list_file = os.path.normpath(os.path.join(dataset_dir, dataset_data.label_list))
        label_list = os.path.relpath(label_list_file, output_dir)
        dataset_train = os.path.normpath(os.path.join(dataset_dir, dataset_data.datasets['training']))
        dataset_val = os.path.normpath(os.path.join(dataset_dir, dataset_data.datasets['validation']))
        datasets = Map({
            'training': os.path.relpath(dataset_train, output_dir),
            'validation': os.path.relpath(dataset_val, output_dir),
        })
        args = Map({
            'training_name': output_file_name,
            'output_dir': output_dir,
            'batch_size': int(self.args_batch_size.currentText()),
            'gpus': gpus,
            'epochs': int(self.args_epochs.value()),
        })
    
        Training.create_training_config(output_file, network_idx, dataset_data.format, label_list, datasets, args)

        # Pass full paths to training function
        datasets.training = os.path.normpath(os.path.join(output_dir, datasets.training))
        datasets.validation = os.path.normpath(os.path.join(output_dir, datasets.validation))
        args.classes_list = label_list_file

        # Start training
        training_func = getattr(self, func_name)
        training_func(datasets, args)

    def cancel_btn_clicked(self):
        self.close()

    def dataset_file_browse_btn_clicked(self):
        filters = _('Dataset file') + ' (*{})'.format(Export.config('config_file_extension'))
        dataset_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select dataset file'), self.parent.exportState.lastFile, filters)
        if dataset_file:
            dataset_file = os.path.normpath(dataset_file)
            self.dataset_file.setText(dataset_file)

    def output_browse_btn_clicked(self):
        last_file = self.parent.settings.value('training/last_output_file', '')
        logger.debug('Restored value "{}" for setting training/last_output_file'.format(last_file))
        filters = _('Training file') + ' (*{})'.format(Training.config('config_file_extension'))
        output_file, selected_filter = QtWidgets.QFileDialog.getSaveFileName(self, _('Save output file as'), last_file, filters)
        if output_file:
            output_file = os.path.normpath(output_file)
            self.parent.settings.setValue('training/last_output_file', output_file)
            self.output_file.setText(output_file)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    def yolov3(self, datasets, args):
        self.progress.setMaximum(args.epochs + 4)
        self.progress.setLabelText(_('Initializing training thread ...'))
        self.progress.setValue(0)

        network = NetworkYoloV3()
        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = TrainingObject(worker, network.training, self.error_training_progress, network.abort, self.update_training_progress, self.finish_training_progress)
        network.init_training(self.worker_object, args.training_name, args.output_dir, args.classes_list, datasets.training, 
            validate_dataset=datasets.validation, batch_size=args.batch_size, gpus=args.gpus, epochs=args.epochs)

        output_file = self.output_file.text()
        data = {
            'architecture': network.getArchitectureFilename(),
            'weights': network.getWeightsFilename(),
            'args': network.getArgs(),
        }
        Training.update_training_config(output_file, data)

        self.progress.canceled.disconnect()
        self.progress.canceled.connect(self.abort_training_progress)
        worker.addObject(self.worker_object)
        worker.start()

    def update_training_progress(self, msg=None, value=None):
        if self.progress.wasCanceled():
            return
        if msg is not None:
            self.progress.setLabelText(msg)
        if value is not None:
            self.progress.setValue(value)

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

