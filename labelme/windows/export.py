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


class ExportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super(ExportWindow, self).__init__(parent)
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
        export_func(data_folder, export_folder, label_files)

        if self.progress.wasCanceled():
            self.progress.close()
            return

        self.progress.close()

        self.parent.lastExportDir = export_folder

        mb = QtWidgets.QMessageBox
        mb.information(self, _('Export'), _('Dataset has been exported successfully to: {}').format(export_folder))
        self.close()

    def cancel_btn_clicked(self):
        self.close()

    def dataset_browse_btn_clicked(self):
        data_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select data folder'))
        self.data_folder.setText(data_folder)

    def export_browse_btn_clicked(self):
        export_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select export folder'))
        self.export_folder.setText(export_folder)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    # Export functions
    def _imagerecord(self, data_folder, export_folder, label_files):
        num_label_files = len(label_files)
        self.progress.setMaximum(num_label_files * 2)

        # First, generate class list
        label_list_file = export.make_label_list(export_folder, label_files, self.progress)

        # Create lst file
        lst_file = export.make_lst_file(export_folder, label_files, label_list_file, self.progress)

        if self.progress.wasCanceled():
            return

        # Then, create rec file from lst file
        rec_file = export.im2rec(lst_file, data_folder, progress=self.progress, num_label_files=num_label_files, no_shuffle=False, pass_through=True, pack_label=True)

        self.parent.lastExportFile = rec_file

    