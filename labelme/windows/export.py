from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import export
import os
import sys
import importlib
import subprocess

class Export():

    _filters = None
    _filter2format = None
    _extension2format = None

    @staticmethod
    def config(key = None):
        config = {
            'default_dataset_name': 'dataset',
            'formats': {
                '_imagerecord': _('ImageRecord'),
            },
            'extensions': {
                '_imagerecord': '.rec',
            }
        }
        if key is not None:
            if key in config:
                return config[key]
        return config

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
        self.lastDir = None
        self.lastFileTrain = None
        self.lastFileVal = None
        self.lastExtension = None


class ExportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super().__init__(parent)
        self.setWindowTitle(_('Export'))
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

        self.export_folder = QtWidgets.QLineEdit()
        export_browse_btn = QtWidgets.QPushButton(_('Browse'))
        export_browse_btn.clicked.connect(self.export_browse_btn_clicked)

        export_folder_group = QtWidgets.QGroupBox()
        export_folder_group.setTitle(_('Export folder'))
        export_folder_group_layout = QtWidgets.QHBoxLayout()
        export_folder_group.setLayout(export_folder_group_layout)
        export_folder_group_layout.addWidget(self.export_folder)
        export_folder_group_layout.addWidget(export_browse_btn)
        layout.addWidget(export_folder_group)

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

        export_folder = self.export_folder.text()
        if not export_folder or not os.path.isdir(export_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Export'), _('Please enter a valid export folder'))
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

        export_func = getattr(self, func_name)
        export_func(data_folder, export_folder, label_files, validation_ratio)

        if self.progress.wasCanceled():
            self.progress.close()
            return

        self.progress.close()

        self.parent.exportState.lastDir = export_folder

        mb = QtWidgets.QMessageBox
        mb.information(self, _('Export'), _('Dataset has been exported successfully to: {}').format(export_folder))
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
        export_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select export folder'), last_dir)
        if export_folder:
            export_folder = os.path.normpath(export_folder)
            self.parent.settings.setValue('export/last_export_dir', export_folder)
        self.export_folder.setText(export_folder)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    # Export functions
    def _imagerecord(self, data_folder, export_folder, label_files, validation_ratio=0.0):
        num_label_files = len(label_files)
        self.progress.setMaximum(num_label_files * 2)
        num_label_files_train = int(num_label_files * (1.0 - validation_ratio))
        num_label_files_val = int(num_label_files * validation_ratio)

        # First, generate class list
        label_list_file = export.make_label_list(export_folder, label_files, self.progress)

        if self.progress.wasCanceled():
            return

        # Create lst file
        lst_train, lst_val = export.make_lst_file(export_folder, label_files, label_list_file, self.progress, validation_ratio)

        if self.progress.wasCanceled():
            return

        # Then, create rec file from lst file
        rec_file_train = export.lst2rec(lst_train[0], data_folder, progress=self.progress, 
            num_label_files=lst_train[1], pass_through=True, pack_label=True)

        if self.progress.wasCanceled():
            return

        rec_file_val = export.lst2rec(lst_val[0], data_folder, progress=self.progress, 
            num_label_files=lst_val[1], pass_through=True, pack_label=True)

        extension = '.' + str(rec_file_train.split('.')[-1:][0])
        self.parent.exportState.lastFileTrain = rec_file_train
        self.parent.exportState.lastFileVal = rec_file_val
        self.parent.exportState.lastExtension = extension

    