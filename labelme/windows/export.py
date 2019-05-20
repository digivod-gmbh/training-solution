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
from labelme.utils import Worker, ProgressObject, Application
from labelme.utils.map import Map
import labelme.extensions.formats as formats
from labelme.config import Export


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
        self.progress.setMaximum(4 * len(label_files) + 3)
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

        dataset_format = Export.config('objects')[func_name]()
        
        dataset_format.make_label_list(label_list_file, label_files)
        format_idx = func_name
        args = Map({
            'validation_ratio': validation_ratio,
            'test_ratio': 0.0
        })
        Export.create_dataset_config(export_file, format_idx, label_list_file_relative, args)

        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = ProgressObject(worker, dataset_format.export, self.error_export_progress, dataset_format.abort, 
            self.update_export_progress, self.finish_export_progress)
        dataset_format.init_export(self.worker_object, data_folder, export_file, label_files, label_list_file, validation_ratio)

        dataset_file_train = dataset_format.getTrainingFilename(export_dir, export_file_name)
        dataset_file_val = dataset_format.getValidateFilename(export_dir, export_file_name)

        export_dir = os.path.dirname(export_file)
        data = Map({
            'samples': {
                'training': dataset_file_train,
                'validation': dataset_file_val,
            },
            'datasets': {
                'training': os.path.relpath(dataset_file_train, export_dir),
                'validation': os.path.relpath(dataset_file_val, export_dir),
            }
        })
        Export.update_dataset_config(export_file, data)

        extension = '.' + str(dataset_file_train.split('.')[-1:][0])
        self.parent.exportState.lastFileTrain = dataset_file_train
        self.parent.exportState.lastFileVal = dataset_file_val
        self.parent.exportState.lastExtension = extension
        self.parent.exportState.lastFile = export_file

        self.progress.canceled.disconnect()
        self.progress.canceled.connect(self.abort_export_progress)
        worker.addObject(self.worker_object)
        worker.start()

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

    def update_export_progress(self, msg=None, value=None):
        if self.progress.wasCanceled():
            return
        if msg is not None:
            self.progress.setLabelText(msg)
        if value is not None:
            self.progress.setValue(value)

    def abort_export_progress(self):
        self.progress.setLabelText(_('Cancelling ...'))
        self.progress.setMaximum(0)
        self.worker_object.abort()
        worker = Application.getWorker(self.worker_idx)
        worker.wait()
        self.progress.cancel()
        Application.destroyWorker(self.worker_idx)

    def finish_export_progress(self):
        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Export'), _('Dataset has been exported successfully'))
        self.progress.close()
        self.close()

    def error_export_progress(self, e):
        self.progress.cancel()
        mb = QtWidgets.QMessageBox()
        mb.warning(self, _('Export'), _('An error occured during export of dataset'))

    