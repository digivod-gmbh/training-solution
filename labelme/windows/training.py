from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import importlib
import os
import re
import mxnet as mx
import json
import math
import time
import glob
import traceback

from labelme.logger import logger
from labelme.utils import deltree, WorkerDialog, QHLine, confirm, replace_special_chars
from labelme.utils.map import Map
from labelme.utils import WorkerExecutor
from labelme.extensions.formats import *
from labelme.extensions.networks import Network
from labelme.config import MessageType
from labelme.config import Training
from labelme.config.export import Export
from labelme.windows import ExportExecutor, TrainingProgressWindow
from labelme.widgets import HelpLabel, HelpCheckbox, HelpGroupBox


class TrainingWindow(WorkerDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Training'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.NonModal)

        self.dataset_format_init = False
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        tab_dataset = QtWidgets.QWidget()
        tab_dataset_layout = QtWidgets.QVBoxLayout()
        tab_dataset.setLayout(tab_dataset_layout)
        self.tabs.addTab(tab_dataset, _('Dataset'))

        tab_network = QtWidgets.QWidget()
        tab_network_layout = QtWidgets.QVBoxLayout()
        tab_network.setLayout(tab_network_layout)
        self.tabs.addTab(tab_network, _('Network'))
        
        tab_resume = QtWidgets.QWidget()
        tab_resume_layout = QtWidgets.QVBoxLayout()
        tab_resume.setLayout(tab_resume_layout)
        self.tabs.addTab(tab_resume, _('Resume training'))

        # Network Tab

        self.networks = QtWidgets.QComboBox()
        for key, val in Training.config('networks').items():
            self.networks.addItem(val)
        self.networks.currentIndexChanged.connect(self.network_selection_changed)

        network_group = HelpGroupBox('Training_NetworkArchitecture', _('Network'))
        network_group_layout = QtWidgets.QVBoxLayout()
        network_group.widget.setLayout(network_group_layout)
        network_group_layout.addWidget(self.networks)
        tab_network_layout.addWidget(network_group)

        training_defaults = self.parent._config['training_defaults']
        network = self.get_current_network()

        args_epochs_label = HelpLabel('Training_SettingsEpochs', _('Epochs'))
        self.args_epochs = QtWidgets.QSpinBox()
        self.args_epochs.setMinimum(1)
        self.args_epochs.setMaximum(500)
        self.args_epochs.setValue(training_defaults['epochs'])

        default_batch_size = self.get_default_batch_size(network)
        args_batch_size_label = HelpLabel('Training_SettingsBatchSize', _('Batch size'))
        self.args_batch_size = QtWidgets.QSpinBox()
        self.args_batch_size.setMinimum(1)
        self.args_batch_size.setMaximum(100)
        self.args_batch_size.setValue(default_batch_size)

        default_learning_rate = self.get_default_learning_rate(network)
        args_learning_rate_label = HelpLabel('Training_SettingsLearningRate', _('Learning rate'))
        self.args_learning_rate = QtWidgets.QDoubleSpinBox()
        self.args_learning_rate.setMinimum(1e-7)
        self.args_learning_rate.setMaximum(1.0)
        self.args_learning_rate.setSingleStep(1e-7)
        self.args_learning_rate.setDecimals(7)
        self.args_learning_rate.setValue(default_learning_rate)

        args_early_stop_epochs_label = HelpLabel('Training_SettingsEarlyStop', _('Early stop epochs'))
        self.args_early_stop_epochs = QtWidgets.QSpinBox()
        self.args_early_stop_epochs.setMinimum(0)
        self.args_early_stop_epochs.setMaximum(100)
        self.args_early_stop_epochs.setValue(training_defaults['early_stop_epochs'])

        self.gpu_label_text = _('GPU')
        args_gpus_label = QtWidgets.QLabel(_('GPUs'))
        no_gpus_available_label = QtWidgets.QLabel(_('No GPUs available'))
        self.gpus = mx.test_utils.list_gpus() # ['0']
        self.gpu_checkboxes = []
        for i in self.gpus:
            checkbox = QtWidgets.QCheckBox('{} {}'.format(self.gpu_label_text, i))
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
        settings_group_layout.addWidget(args_early_stop_epochs_label, 3, 0)
        settings_group_layout.addWidget(self.args_early_stop_epochs, 3, 1)

        settings_group_layout.addWidget(args_gpus_label, 4, 0)
        if len(self.gpu_checkboxes) > 0:
            row = 4
            for i, checkbox in enumerate(self.gpu_checkboxes):
                settings_group_layout.addWidget(checkbox, row, 1)
                row += 1
        else:
            settings_group_layout.addWidget(no_gpus_available_label, 4, 1)
        tab_network_layout.addWidget(settings_group)

        # Dataset Tab

        image_list = self.parent.imageList
        show_dataset_create = len(image_list) > 0
        self.create_dataset_checkbox = HelpCheckbox('Training_CreateDataset', _('Create dataset from opened images'))
        self.create_dataset_checkbox.widget.setChecked(show_dataset_create)
        tab_dataset_layout.addWidget(self.create_dataset_checkbox)

        validation_label = HelpLabel('Training_ValidationRatio', _('Validation ratio'))
        self.validation = QtWidgets.QSpinBox()
        self.validation.setValue(10)
        self.validation.setMinimum(0)
        self.validation.setMaximum(90)
        self.validation.setFixedWidth(50)
        validation_description_label = QtWidgets.QLabel(_('% of dataset'))

        dataset_name_label = QtWidgets.QLabel(_('Dataset name'))
        self.dataset_name = QtWidgets.QLineEdit()

        self.create_dataset_group = QtWidgets.QGroupBox()
        self.create_dataset_group.setTitle(_('Create dataset'))
        create_dataset_group_layout = QtWidgets.QGridLayout()
        self.create_dataset_group.setLayout(create_dataset_group_layout)
        create_dataset_group_layout.addWidget(dataset_name_label, 0, 0, 1, 2)
        create_dataset_group_layout.addWidget(self.dataset_name, 1, 0, 1, 2)
        create_dataset_group_layout.addWidget(validation_label, 2, 0, 1, 2)
        create_dataset_group_layout.addWidget(self.validation, 3, 0)
        create_dataset_group_layout.addWidget(validation_description_label, 3, 1)

        formats_label = HelpLabel('Training_DatasetFormat', _('Format'))
        self.formats = QtWidgets.QComboBox()
        for key, val in Export.config('formats').items():
            self.formats.addItem(val)
        self.formats.setCurrentIndex(0)
        self.formats.currentTextChanged.connect(self.on_format_change)
        self.selected_format = list(Export.config('formats').keys())[0]

        train_dataset_label = HelpLabel('Training_TrainingDataset', _('Training dataset'))
        self.train_dataset_folder = QtWidgets.QLineEdit()
        train_dataset_folder_browse_btn = QtWidgets.QPushButton(_('Browse'))
        train_dataset_folder_browse_btn.clicked.connect(self.train_dataset_folder_browse_btn_clicked)

        val_label_text = '{} ({})'.format(_('Validation dataset'), _('optional'))
        val_dataset_label = HelpLabel('Training_ValidationDataset', val_label_text)
        self.val_dataset_folder = QtWidgets.QLineEdit()
        val_dataset_folder_browse_btn = QtWidgets.QPushButton(_('Browse'))
        val_dataset_folder_browse_btn.clicked.connect(self.val_dataset_folder_browse_btn_clicked)

        self.dataset_folder_group = QtWidgets.QGroupBox()
        self.dataset_folder_group.setTitle(_('Use dataset file(s)'))
        dataset_folder_group_layout = QtWidgets.QGridLayout()
        self.dataset_folder_group.setLayout(dataset_folder_group_layout)
        dataset_folder_group_layout.addWidget(formats_label, 0, 0, 1, 2)
        dataset_folder_group_layout.addWidget(self.formats, 1, 0, 1, 2)
        dataset_folder_group_layout.addWidget(train_dataset_label, 2, 0, 1, 2)
        dataset_folder_group_layout.addWidget(self.train_dataset_folder, 3, 0)
        dataset_folder_group_layout.addWidget(train_dataset_folder_browse_btn, 3, 1)
        dataset_folder_group_layout.addWidget(val_dataset_label, 4, 0, 1, 2)
        dataset_folder_group_layout.addWidget(self.val_dataset_folder, 5, 0)
        dataset_folder_group_layout.addWidget(val_dataset_folder_browse_btn, 5, 1)

        tab_dataset_layout.addWidget(self.create_dataset_group)
        tab_dataset_layout.addWidget(self.dataset_folder_group)

        if show_dataset_create:
            self.dataset_folder_group.hide()
        else:
            self.create_dataset_checkbox.hide()
            self.create_dataset_group.hide()

        self.create_dataset_checkbox.widget.toggled.connect(lambda: self.switch_visibility(self.create_dataset_group, self.dataset_folder_group))

        self.output_folder = QtWidgets.QLineEdit()
        self.output_folder.setText(os.path.join(project_folder, self.parent._config['project_training_folder']))
        # self.output_folder.setReadOnly(True)
        # output_browse_btn = QtWidgets.QPushButton(_('Browse'))
        # output_browse_btn.clicked.connect(self.output_browse_btn_clicked)

        training_name_label = HelpLabel('Training_TrainingName', _('Training name'))
        self.training_name = QtWidgets.QLineEdit()

        output_folder_group = QtWidgets.QGroupBox()
        output_folder_group.setTitle(_('Output folder'))
        output_folder_group_layout = QtWidgets.QGridLayout()
        output_folder_group.setLayout(output_folder_group_layout)
        # output_folder_group_layout.addWidget(self.output_folder, 0, 0, 1, 2)
        # output_folder_group_layout.addWidget(output_browse_btn, 0, 2)
        output_folder_group_layout.addWidget(training_name_label, 1, 0, 1, 3)
        output_folder_group_layout.addWidget(self.training_name, 2, 0, 1, 3)
        tab_dataset_layout.addWidget(output_folder_group)

        # Resume Tab

        self.resume_training_checkbox = HelpCheckbox('Training_Resume', _('Resume previous training'))
        self.resume_training_checkbox.widget.setChecked(False)
        tab_resume_layout.addWidget(self.resume_training_checkbox)

        self.resume_group = QtWidgets.QWidget()
        resume_group_layout = QtWidgets.QVBoxLayout()
        self.resume_group.setLayout(resume_group_layout)
        tab_resume_layout.addWidget(self.resume_group)

        self.resume_group.hide()
        self.resume_training_checkbox.widget.toggled.connect(self.toggle_resume_training_checkbox)

        self.resume_folder = QtWidgets.QLineEdit()
        self.resume_folder.setText('')
        self.resume_folder.setReadOnly(True)
        resume_browse_btn = QtWidgets.QPushButton(_('Browse'))
        resume_browse_btn.clicked.connect(self.resume_browse_btn_clicked)

        resume_folder_group = QtWidgets.QGroupBox()
        resume_folder_group.setTitle(_('Training directory'))
        resume_folder_group_layout = QtWidgets.QHBoxLayout()
        resume_folder_group.setLayout(resume_folder_group_layout)
        resume_folder_group_layout.addWidget(self.resume_folder)
        resume_folder_group_layout.addWidget(resume_browse_btn)
        resume_group_layout.addWidget(resume_folder_group)

        self.resume_file = QtWidgets.QComboBox()
        self.resume_file_group = QtWidgets.QGroupBox()
        self.resume_file_group.setTitle(_('Resume after'))
        resume_file_group_layout = QtWidgets.QHBoxLayout()
        self.resume_file_group.setLayout(resume_file_group_layout)
        resume_file_group_layout.addWidget(self.resume_file)
        resume_group_layout.addWidget(self.resume_file_group)
        self.resume_file_group.hide()

        tab_dataset_layout.addStretch()
        tab_network_layout.addStretch()
        tab_resume_layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox()
        self.training_btn = button_box.addButton(_('Start Training'), QtWidgets.QDialogButtonBox.AcceptRole)
        self.training_btn.clicked.connect(self.training_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

        h = self.sizeHint().height()
        self.resize(500, h)

    def switch_visibility(self, first, second):
        if first.isVisible():
            first.hide()
            second.show()
        else:
            second.hide()
            first.show()

    def toggle_resume_training_checkbox(self):
        if self.resume_group.isVisible():
            self.resume_group.hide()
            self.training_btn.setText(_('Start training'))
            self.resume_folder.setText('')
            self.resume_file_group.hide()
            self.resume_file.clear()
        else:
            self.resume_group.show()

    def on_format_change(self, value):
        formats = Export.config('formats')
        inv_formats = Export.invertDict(formats)
        if value in inv_formats:
            self.selected_format = inv_formats[value]
            logger.debug('Selected dataset format: {}'.format(self.selected_format))
        else:
            logger.debug('Dataset format not found: {}'.format(value))

    def network_selection_changed(self):
        network = self.get_current_network()
        default_batch_size = self.get_default_batch_size(network)
        self.args_batch_size.setValue(default_batch_size)
        learning_rate = self.get_default_learning_rate(network)
        self.args_learning_rate.setValue(learning_rate)

    def get_current_network(self):
        try:
            selected_network = self.networks.currentText()
            networks = Training.config('networks')
            func_name = None
            for key in networks:
                if selected_network in networks[key]:
                    func_name = key
                    break
            if func_name is not None:
                network = Training.config('objects')[func_name]()
                return network
        except Exception as e:
            logger.error(traceback.format_exc())
            return None

    def get_current_network_key(self):
        try:
            selected_network = self.networks.currentText()
            networks = Training.config('networks')
            func_name = None
            for key in networks:
                if selected_network in networks[key]:
                    return key
        except Exception as e:
            logger.error(traceback.format_exc())
            return None

    def set_current_network(self, network_key):
        try:
            networks = Training.config('networks')
            network_text = networks[network_key]
            for i in range(self.networks.count()):
                text = self.networks.itemText(i)
                if text == network_text:
                    self.networks.setCurrentIndex(i)
                    break
        except Exception as e:
            logger.error(traceback.format_exc())

    def set_current_format(self, format_key):
        try:
            formats = Export.config('formats')
            format_text = formats[format_key]
            for i in range(self.networks.count()):
                text = self.formats.itemText(i)
                if text == format_text:
                    self.formats.setCurrentIndex(i)
                    break
        except Exception as e:
            logger.error(traceback.format_exc())


    def get_default_learning_rate(self, network):
        learning_rate = network.getDefaultLearningRate()
        return learning_rate

    def get_default_batch_size(self, network):
        try:
            selected_network = self.networks.currentText()
            network_size_base, network_size_per_batch = network.getGpuSizes()

            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu = gpus[0]

            # Estimate best possible batch size
            # Always take 1GB off to have memory left for peaks
            batch_size = int(math.floor((gpu.memoryFree - network_size_base) / network_size_per_batch))
            estimated_memory = gpu.memoryUsed + network_size_base + batch_size * network_size_per_batch
            logger.debug('Estimating batch size: GPU {} (ID:{}) uses {}MB of {}MB. With {} ({}MB, {}MB) the estimated GPU usage is {}MB at a batch size of {}'
            .format(gpu.name, gpu.id, gpu.memoryUsed, gpu.memoryTotal, selected_network, network_size_base, network_size_per_batch, estimated_memory, batch_size))

            return batch_size

        except Exception as e:
            logger.error(traceback.format_exc())

    def cancel_btn_clicked(self):
        self.close()

    def resume_browse_btn_clicked(self):
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        training_folder = os.path.join(project_folder, self.parent._config['project_training_folder'])
        training_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select training directory'), training_folder)
        if training_folder:
            training_folder = os.path.normpath(training_folder)
            self.resume_folder.setText(training_folder)
            # Prepare to resume training
            self.prepare_resume_training(training_folder)

    def prepare_resume_training(self, training_folder):
        try:
            config_file = os.path.join(training_folder, Training.config('config_file'))
            json_data = {}
            with open(config_file, 'r') as f:
                json_data = json.load(f)

            # Dataset tab
            self.create_dataset_checkbox.widget.setChecked(False)
            self.set_current_format(json_data['dataset'])
            self.train_dataset_folder.setText(json_data['args']['train_dataset'])
            if json_data['args']['validate_dataset']:
                self.val_dataset_folder.setText(json_data['args']['validate_dataset'])
            else:
                self.val_dataset_folder.setText('')
            output_folder = os.path.normpath(os.path.join(training_folder, '..'))
            self.output_folder.setText(output_folder)
            self.training_name.setText(json_data['args']['training_name'] + '_resume')

            # Network tab
            self.set_current_network(json_data['network'])
            self.args_epochs.setValue(json_data['args']['epochs'])
            self.args_batch_size.setValue(json_data['args']['batch_size'])
            self.args_learning_rate.setValue(json_data['args']['learning_rate'])
            self.args_early_stop_epochs.setValue(json_data['args']['early_stop_epochs'])
            gpus = json_data['args']['gpus'].split(',')
            l = len(self.gpu_label_text) + 1
            for box in self.gpu_checkboxes:
                checked = box.text()[l:] in gpus
                box.setChecked(checked)

            # Resume
            self.resume_file.clear()
            params_pattern = '{}_*_*.params'.format(json_data['args']['save_prefix'])
            params_path = os.path.join(training_folder, params_pattern)
            for params_file in reversed(glob.glob(params_path)):
                parts = os.path.splitext(params_file)[0].split('_')
                epoch = int(parts[-2]) + 1
                accuracy = float(parts[-1])
                label = '{} {} ({}={})'.format(_('Epoch'), epoch, _('Accuracy'), accuracy)
                self.resume_file.addItem(label, (params_file, epoch))
            self.resume_file.setCurrentIndex(0)
            self.resume_file_group.show()

            self.training_btn.setText(_('Resume training'))
            
        except Exception as e:
            logger.error(traceback.format_exc())
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Training'), _('Applying config of previous training failed'))
            self.resume_folder.setText('')

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

    # def output_browse_btn_clicked(self):
    #     project_folder = self.parent.settings.value('settings/project/folder', '')
    #     logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
    #     training_folder = os.path.join(project_folder, self.parent._config['project_training_folder'])
    #     output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), training_folder)
    #     if output_folder:
    #         output_folder = os.path.normpath(output_folder)
    #         self.output_folder.setText(output_folder)

    def training_btn_clicked(self):
        create_dataset = self.create_dataset_checkbox.widget.isChecked()
        self.dataset_export_data = {}
        if create_dataset:
            self.export_before_training()
        else:
            self.start_training()

    def export_before_training(self):
        training_defaults = self.parent._config['training_defaults']
        selected_format = training_defaults['dataset_format']
        dataset_name = replace_special_chars(self.dataset_name.text())

        data_folder = None
        if self.parent.lastOpenDir is not None:
            data_folder = self.parent.lastOpenDir
        if data_folder is None:
            mb = QtWidgets.QMessageBox()
            mb.warning(self, _('Training'), _('Please open a folder with images first'))
            return

        all_labels = []
        for i in range(len(self.parent.uniqLabelList)):
            all_labels.append(self.parent.uniqLabelList.item(i).text())
        if len(all_labels) == 0:
            mb = QtWidgets.QMessageBox()
            mb.warning(self, _('Training'), _('No labels found in dataset'))
            return

        project_folder = self.parent.settings.value('settings/project/folder', '')
        project_dataset_folder = self.parent._config['project_dataset_folder']
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        export_folder = os.path.join(project_folder, project_dataset_folder)

        validation_ratio = int(self.validation.text()) / 100.0

        self.dataset_export_data = {
            'dataset_name': dataset_name,
            'format': selected_format,
            'output_folder': os.path.join(export_folder, dataset_name),
            'validation_ratio': validation_ratio,
        }
        self.selected_format = selected_format

        data = {
            'data_folder': data_folder,
            'export_folder': export_folder,
            'selected_labels': all_labels,
            'validation_ratio': validation_ratio,
            'dataset_name': dataset_name,
            'max_num_labels': Export.config('limits')['max_num_labels'],
            'selected_format': Export.config('formats')[selected_format],
        }

        # Execution
        executor = ExportExecutor(data)
        self.run_thread(executor, self.start_training)

    def start_training(self):
        training_defaults = self.parent._config['training_defaults']

        network_key = self.get_current_network_key()
        epochs = self.args_epochs.value()

        resume_training_file = ''
        resume_epoch = 0
        if self.resume_training_checkbox.widget.isChecked() and self.resume_file.count() > 0:
            idx = self.resume_file.currentIndex()
            resume_training_file, resume_epoch = self.resume_file.itemData(idx)
            epochs += resume_epoch
        
        # Data
        data = {
            'create_dataset': self.create_dataset_checkbox.widget.isChecked(),
            'resume_training': resume_training_file,
            'start_epoch': resume_epoch,
            'dataset_export_data': self.dataset_export_data,
            'train_dataset': self.train_dataset_folder.text(),
            'val_dataset': self.val_dataset_folder.text(),
            'output_folder': self.output_folder.text(),
            'selected_format': self.selected_format,
            'training_name': self.training_name.text(),
            'network': network_key,
            'gpu_checkboxes': self.gpu_checkboxes,
            'args_epochs': epochs,
            'args_batch_size': self.args_batch_size.value(),
            'args_learning_rate': self.args_learning_rate.value(),
            'args_early_stop_epochs': self.args_early_stop_epochs.value(),
        }

        # Preprocess data
        mb = QtWidgets.QMessageBox()
        create_dataset = data['create_dataset']
        if create_dataset:
            export_data = data['dataset_export_data']
            format_name = export_data['format']
            dataset_format = Export.config('objects')[format_name]()
            output_folder = export_data['output_folder']
            train_file = dataset_format.getOutputFileName('train')
            train_dataset = os.path.join(output_folder, train_file)
            data['train_dataset'] = train_dataset
            validation_ratio = export_data['validation_ratio']
            if validation_ratio > 0:
                val_file = dataset_format.getOutputFileName('val')
                val_dataset = os.path.join(output_folder, val_file)
            else:
                # Validation dataset is optional
                val_dataset = False
            data['val_dataset'] = val_dataset

        else:
            train_dataset = data['train_dataset']
            is_train_dataset_valid = True
            if not train_dataset:
                is_train_dataset_valid = False
            train_dataset = os.path.normpath(train_dataset)
            if not (os.path.isdir(train_dataset) or os.path.isfile(train_dataset)):
                is_train_dataset_valid = False
            if not is_train_dataset_valid:
                mb.warning(self, _('Training'), _('Please select a valid training dataset'))
                return
            data['train_dataset'] = train_dataset

            val_dataset = data['val_dataset']
            is_val_dataset_valid = True
            if not val_dataset:
                is_val_dataset_valid = False
            val_dataset = os.path.normpath(val_dataset)
            if not (os.path.isdir(val_dataset) or os.path.isfile(val_dataset)):
                is_val_dataset_valid = False
            if not is_val_dataset_valid:
                # Validation dataset is optional
                val_dataset = False
            data['val_dataset'] = val_dataset

        if val_dataset and val_dataset == train_dataset:
            mb.warning(self, _('Training'), _('Training and validation dataset are equal. Please use different datasets, as validation results are useless otherwise.'))
            return

        output_folder = os.path.normpath(data['output_folder'])
        training_name = data['training_name']
        training_name = replace_special_chars(training_name)
        data['training_name'] = training_name

        if not training_name:
            mb.warning(self, _('Training'), _('Please enter a valid training name'))
            return
        
        output_folder = os.path.join(output_folder, training_name)
        data['output_folder'] = output_folder
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        elif len(os.listdir(output_folder)) > 0 and resume_training_file:
            mb.warning(self, _('Training'), _('The selected output directory "{}" is not empty. Please select a different directory.').format(output_folder))
            return
        elif len(os.listdir(output_folder)) > 0 and not resume_training_file:
            msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(output_folder)
            result = confirm(self, _('Training'), msg, MessageType.Warning)
            if result:
                deltree(output_folder)
                time.sleep(0.5) # wait for deletion to be finished
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                return

        if not os.path.isdir(output_folder):
            mb.warning(self, _('Training'), _('The selected output directory "{}" could not be created').format(output_folder))
            return

        # Open new window for training progress
        trainingWin = TrainingProgressWindow(self)
        trainingWin.show()
        trainingWin.start_training(data)
        self.close()

    def finish_training(self):
        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Training'), _('Network has been trained successfully'))
        self.close()

