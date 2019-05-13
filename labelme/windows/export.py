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

class ExportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):

        self.config = {
            'formats': {
                _('ImageRecord'): '_imagerecord'
            }
        }
        self.parent = parent

        super(ExportWindow, self).__init__(parent)
        self.setWindowTitle(_('Export'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.formats = QtWidgets.QComboBox()
        for key in self.config['formats']:
            self.formats.addItem(key)

        format_group = QtWidgets.QGroupBox()
        format_group.setTitle(_('Format'))
        format_group_layout = QtWidgets.QVBoxLayout()
        format_group.setLayout(format_group_layout)
        format_group_layout.addWidget(self.formats)
        layout.addWidget(format_group)

        self.dataset_folder = QtWidgets.QLineEdit()
        dataset_browse_btn = QtWidgets.QPushButton(_('Browse'))
        dataset_browse_btn.clicked.connect(self.dataset_browse_btn_clicked)

        dataset_folder_group = QtWidgets.QGroupBox()
        dataset_folder_group.setTitle(_('Dataset folder'))
        dataset_folder_group_layout = QtWidgets.QHBoxLayout()
        dataset_folder_group.setLayout(dataset_folder_group_layout)
        dataset_folder_group_layout.addWidget(self.dataset_folder)
        dataset_folder_group_layout.addWidget(dataset_browse_btn)
        layout.addWidget(dataset_folder_group)

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

        # Get number of label files
        dataset_folder = self.dataset_folder.text()
        label_files = []
        for root, dirs, files in os.walk(dataset_folder):
            for f in files:
                if LabelFile.is_label_file(f):
                    label_files.append(os.path.normpath(os.path.join(dataset_folder, f)))
        num_label_files = len(label_files)
        logger.debug('{} label files found in dataset folder "{}"'.format(num_label_files, dataset_folder))

        self.progress = QtWidgets.QProgressDialog(_('Exporting dataset ...'), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setValue(0)
        self.progress.show()

        export_folder = self.export_folder.text()

        key = self.formats.currentText()
        if key in self.config['formats']:
            func_name = self.config['formats'][key]
            export_func = getattr(self, func_name)
            export_func(dataset_folder, export_folder, label_files)

        self.progress.close()

        mb = QtWidgets.QMessageBox
        mb.information(self, _('Export'), _('Dataset has been exported successfully to: {}').format(export_folder))
        self.close()

    def cancel_btn_clicked(self):
        self.close()

    def dataset_browse_btn_clicked(self):
        dataset_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select dataset folder'))
        self.dataset_folder.setText(dataset_folder)

    def export_browse_btn_clicked(self):
        export_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select export folder'))
        self.export_folder.setText(export_folder)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

    # Export functions
    def _imagerecord(self, dataset_folder, export_folder, label_files):

        num_label_files = len(label_files)
        self.progress.setMaximum(num_label_files * 2)

        # First, create lst file
        lst_file = export.make_lst_file(export_folder, label_files, self.progress)

        # Then, create rec file from lst file
        export.im2rec(lst_file, dataset_folder, progress=self.progress, num_label_files=num_label_files, no_shuffle=False, pass_through=True, pack_label=True)

    