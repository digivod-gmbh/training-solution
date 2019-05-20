from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import os
import sys
import json
import importlib
import subprocess

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import export
from labelme.utils.map import Map


class Export():

    _filters = None
    _filter2format = None
    _extension2format = None

    @staticmethod
    def config(key = None):
        config = {
            'config_file_extension': '.dataset',
            'formats': {
                'imagerecord': _('ImageRecord'),
            },
            'extensions': {
                'imagerecord': '.rec',
            }
        }
        if key is not None:
            if key in config:
                return config[key]
            return None
        return config

    @staticmethod
    def create_dataset_config(config_file, dataset_format, label_list, args):
        data = {
            'format': dataset_format,
            'label_list': label_list,
            'args': args
        }
        logger.debug('Create dataset config: {}'.format(data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved dataset config in file: {}'.format(config_file))

    @staticmethod
    def update_dataset_config(config_file, new_data):
        old_data = {}
        with open(config_file, 'r') as f:
            old_data = json.loads(f.read())
            logger.debug('Loaded dataset config: {}'.format(old_data))
        data = old_data.copy()
        data.update(new_data)
        logger.debug('Update dataset config: {}'.format(new_data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved dataset config in file: {}'.format(config_file))

    @staticmethod
    def read_dataset_config(config_file):
        data = {}
        with open(config_file, 'r') as f:
            data = json.loads(f.read())
            logger.debug('Read dataset config: {}'.format(data))
        return Map(data)


    @staticmethod
    def filters():
        Export.init_filters()
        return Export._filters

    @staticmethod
    def filter2format(key):
        Export.init_filters()
        if key in Export._filter2format:
            return Export._filter2format[key]
        return None

    @staticmethod
    def extension2format(key):
        Export.init_filters()
        if key in Export._extension2format:
            return Export._extension2format[key]
        return None

    @staticmethod
    def init_filters():
        if Export._filters is None or Export._filter2format is None:
            formats = Export.config('formats')
            extensions = Export.config('extensions')
            filters = []
            Export._filter2format = {}
            Export._extension2format = {}
            for key in formats:
                f = '{} (*{})'.format(formats[key], extensions[key])
                filters.append(f)
                Export._filter2format[f] = formats[key]
                ext = extensions[key]
                Export._extension2format[ext] = formats[key]
            Export._filters = ';;'.join(filters)


class ExportState():

    def __init__(self):
        self.lastFile = None
        self.lastFileTrain = None
        self.lastFileVal = None
        self.lastExtension = None


class ExportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super().__init__(parent)
        self.setWindowTitle(_('Export dataset'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.formats = QtWidgets.QComboBox()
        for key, val in Export.config('formats').items():
            self.formats.addItem(val)

        format_group = QtWidgets.QGroupBox()
        format_group.setTitle(_('Format'))
        format_group_layout = QtWidgets.QVBoxLayout()
        format_group.setLayout(format_group_layout)
        format_group_layout.addWidget(self.formats)
        layout.addWidget(format_group)

        self.data_folder = QtWidgets.QLineEdit()
        if self.parent.lastOpenDir is not None:
            self.data_folder.setText(self.parent.lastOpenDir)
        dataset_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_browse_btn.clicked.connect(self.dataset_browse_btn_clicked)

        data_folder_group = QtWidgets.QGroupBox()
        data_folder_group.setTitle(_('Data folder'))
        data_folder_group_layout = QtWidgets.QHBoxLayout()
        data_folder_group.setLayout(data_folder_group_layout)
        data_folder_group_layout.addWidget(self.data_folder)
        data_folder_group_layout.addWidget(dataset_browse_btn)
        layout.addWidget(data_folder_group)

        self.export_file = QtWidgets.QLineEdit()
        export_browse_btn = QtWidgets.QPushButton(_('Browse'))
        export_browse_btn.clicked.connect(self.export_browse_btn_clicked)

        export_file_group = QtWidgets.QGroupBox()
        export_file_group.setTitle(_('Export folder'))
        export_file_group_layout = QtWidgets.QHBoxLayout()
        export_file_group.setLayout(export_file_group_layout)
        export_file_group_layout.addWidget(self.export_file)
        export_file_group_layout.addWidget(export_browse_btn)
        layout.addWidget(export_file_group)

        self.validation = QtWidgets.QSpinBox()
        self.validation.setValue(10)
        self.validation.setMinimum(0)
        self.validation.setMaximum(90)
        self.validation.setFixedWidth(50)
        validation_label = QtWidgets.QLabel(_('% of dataset'))

        validation_group = QtWidgets.QGroupBox()
        validation_group.setTitle(_('Validation'))
        validation_group_layout = QtWidgets.QGridLayout()
        validation_group.setLayout(validation_group_layout)
        validation_group_layout.addWidget(self.validation, 0, 0)
        validation_group_layout.addWidget(validation_label, 0, 1)
        layout.addWidget(validation_group)

        button_box = QtWidgets.QDialogButtonBox()
        export_btn = button_box.addButton(_('Export'), QtWidgets.QDialogButtonBox.AcceptRole)
        export_btn.clicked.connect(self.export_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def export_btn_clicked(self):
        data_folder = self.data_folder.text()
        if not data_folder or not os.path.isdir(data_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Export'), _('Please enter a valid data folder'))
            return

        export_file = self.export_file.text()
        export_file_name = os.path.splitext(os.path.basename(export_file))[0]
        export_dir = os.path.dirname(export_file)
        if not export_dir or not os.path.isdir(export_dir):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Export'), _('Please enter a valid export folder'))
            return

        if len(os.listdir(export_dir)) > 0:
            mb = QtWidgets.QMessageBox
            msg = _('The selected output directory "{}" is not empty. Containing files could be overwritten. Are you sure to continue?').format(export_dir)
            clicked_btn = mb.warning(self, _('Export'), msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if clicked_btn == QtWidgets.QMessageBox.No:
                return

        label_files = []
        for root, dirs, files in os.walk(data_folder):
            for f in files:
                if LabelFile.is_label_file(f):
                    label_files.append(os.path.normpath(os.path.join(data_folder, f)))
        num_label_files = len(label_files)
        logger.debug('Found {} label files in dataset folder "{}"'.format(num_label_files, data_folder))

        validation_ratio = int(self.validation.value()) / 100.0

        self.progress = QtWidgets.QProgressDialog(_('Exporting dataset ...'), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setValue(0)
        self.progress.show()

        val = self.formats.currentText()
        formats = Export.config('formats')
        func_name = None
        for key in formats:
            if val in formats[key]:
                func_name = key
        
        if func_name is None:
            logger.error('Export format {} could not be found'.format(val))
            return

        label_list_file = os.path.normpath(os.path.join(export_dir, '{}.labels'.format(export_file_name)))
        label_list_file_relative = os.path.relpath(label_list_file, export_dir)

        export.make_label_list(label_list_file, label_files)
        format_idx = func_name
        args = Map({
            'validation_ratio': validation_ratio,
            'test_ratio': 0.0
        })
        Export.create_dataset_config(export_file, format_idx, label_list_file_relative, args)

        export_func = getattr(self, func_name)
        export_func(data_folder, export_file, label_files, label_list_file, validation_ratio)

        if self.progress.wasCanceled():
            self.progress.close()
            return

        self.progress.close()

        self.parent.exportState.lastFile = export_file

        mb = QtWidgets.QMessageBox
        mb.information(self, _('Export'), _('Dataset has been exported successfully to: {}').format(export_file))
        self.close()

    def cancel_btn_clicked(self):
        self.close()

    def dataset_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('export/last_dataset_dir', '')
        logger.debug('Restored value "{}" for setting export/last_dataset_dir'.format(last_dir))
        data_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select data folder'), last_dir)
        if data_folder:
            data_folder = os.path.normpath(data_folder)
            self.parent.settings.setValue('export/last_dataset_dir', data_folder)
            self.data_folder.setText(data_folder)

    def export_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('export/last_export_dir', '')
        logger.debug('Restored value "{}" for setting export/last_export_dir'.format(last_dir))
        filters = _('Dataset file') + ' (*{})'.format(Export.config('config_file_extension'))
        export_file, selected_filter = QtWidgets.QFileDialog.getSaveFileName(self, _('Save output file as'), last_dir, filters)
        if export_file:
            export_file = os.path.normpath(export_file)
            self.parent.settings.setValue('export/last_export_dir', os.path.dirname(export_file))
            self.export_file.setText(export_file)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    # Export functions
    def imagerecord(self, data_folder, export_file, label_files, label_list_file, validation_ratio=0.0):
        num_label_files = len(label_files)
        self.progress.setMaximum(num_label_files * 2)
        num_label_files_train = int(num_label_files * (1.0 - validation_ratio))
        num_label_files_val = int(num_label_files * validation_ratio)

        # First, create lst file
        lst_train, lst_val = export.make_lst_file(export_file, label_files, label_list_file, self.progress, validation_ratio)

        if self.progress.wasCanceled():
            return

        # Then, create rec file from lst file
        rec_file_train = export.lst2rec(lst_train[0], data_folder, progress=self.progress, 
            num_label_files=lst_train[1], pass_through=True, pack_label=True)

        if self.progress.wasCanceled():
            return

        rec_file_val = export.lst2rec(lst_val[0], data_folder, progress=self.progress, 
            num_label_files=lst_val[1], pass_through=True, pack_label=True)

        export_dir = os.path.dirname(export_file)
        data = Map({
            'samples': {
                'training': lst_train[1],
                'validation': lst_val[1]
            },
            'datasets': {
                'training': os.path.relpath(rec_file_train, export_dir),
                'validation': os.path.relpath(rec_file_val, export_dir),
            }
        })
        Export.update_dataset_config(export_file, data)

        extension = '.' + str(rec_file_train.split('.')[-1:][0])
        self.parent.exportState.lastFileTrain = rec_file_train
        self.parent.exportState.lastFileVal = rec_file_val
        self.parent.exportState.lastExtension = extension

    