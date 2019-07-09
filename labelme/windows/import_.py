from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import os

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import Worker, ProgressObject, Application
from labelme.utils.map import Map
from labelme.extensions.formats import *
from labelme.config import Export


class ImportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super().__init__(parent)
        self.setWindowTitle(_('Import dataset'))
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
        self.data_folder.setReadOnly(True)
        data_browse_btn = QtWidgets.QPushButton(_('Browse'))
        data_browse_btn.clicked.connect(self.data_browse_btn_clicked)

        data_folder_group = QtWidgets.QGroupBox()
        data_folder_group.setTitle(_('Dataset folder'))
        data_folder_group_layout = QtWidgets.QHBoxLayout()
        data_folder_group.setLayout(data_folder_group_layout)
        data_folder_group_layout.addWidget(self.data_folder)
        data_folder_group_layout.addWidget(data_browse_btn)
        layout.addWidget(data_folder_group)

        self.output_folder = QtWidgets.QLineEdit()
        self.output_folder.setReadOnly(True)
        output_browse_btn = QtWidgets.QPushButton(_('Browse'))
        output_browse_btn.clicked.connect(self.output_browse_btn_clicked)

        output_folder_group = QtWidgets.QGroupBox()
        output_folder_group.setTitle(_('Output folder'))
        output_folder_group_layout = QtWidgets.QHBoxLayout()
        output_folder_group.setLayout(output_folder_group_layout)
        output_folder_group_layout.addWidget(self.output_folder)
        output_folder_group_layout.addWidget(output_browse_btn)
        layout.addWidget(output_folder_group)

        button_box = QtWidgets.QDialogButtonBox()
        export_btn = button_box.addButton(_('Import'), QtWidgets.QDialogButtonBox.AcceptRole)
        export_btn.clicked.connect(self.import_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def data_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('import/last_data_dir', '')
        logger.debug('Restored value "{}" for setting import/last_data_dir'.format(last_dir))
        data_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select dataset folder'), last_dir)
        if data_folder:
            data_folder = os.path.normpath(data_folder)
            self.parent.settings.setValue('import/last_data_dir', data_folder)
            self.data_folder.setText(data_folder)

    def output_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('import/last_output_dir', '')
        logger.debug('Restored value "{}" for setting import/last_output_dir'.format(last_dir))
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), last_dir)
        if output_folder:
            output_folder = os.path.normpath(output_folder)
            self.parent.settings.setValue('import/last_output_dir', output_folder)
            self.output_folder.setText(output_folder)

    def import_btn_clicked(self):
        data_folder = self.data_folder.text()
        if not data_folder or not os.path.isdir(data_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('Please enter a valid dataset folder'))
            return

        output_folder = self.output_folder.text()
        if not output_folder or not os.path.isdir(output_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('Please enter a valid output folder'))
            return

        config_file = os.path.join(data_folder, Export.config('config_file'))
        if not os.path.isfile(config_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('No config file found in dataset folder'))
            return

        val = self.formats.currentText()
        formats = Export.config('formats')
        format_name = None
        for key in formats:
            if val in formats[key]:
                format_name = key
        
        if format_name is None:
            logger.error('Import format {} could not be found'.format(val))
            return

        # Dataset
        dataset_format = Export.config('objects')[format_name]()
        if not dataset_format.isValidFormat(data_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('Dataset format could not be recognized'))
            return

        config = dataset_format.loadConfig(config_file)
        label_file = os.path.join(data_folder, Export.config('labels_file'))
        args = Map({
            'config': config,
            'label_file': label_file
        })

        dataset_format.setOutputFolder(output_folder)
        dataset_format.setInputFolder(data_folder)
        dataset_format.setArgs(args)

        self.progress = QtWidgets.QProgressDialog(_('Initializing ...'), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.show()
        self.progress.setMaximum(999)
        self.progress.setLabelText(_('Loading data ...'))
        self.progress.setValue(0)

        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = ProgressObject(worker, dataset_format.import_folder, self.error_import_progress, dataset_format.abort, 
            self.update_import_progress, self.finish_import_progress)
        dataset_format.setThread(self.worker_object)

        self.progress.canceled.disconnect()
        self.progress.canceled.connect(self.abort_import_progress)
        worker.addObject(self.worker_object)
        worker.start()

    def cancel_btn_clicked(self):
        self.close()

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

    def update_import_progress(self, msg=None, value=None):
        if self.progress.wasCanceled():
            return
        if msg:
            self.progress.setLabelText(msg)
        if value is not None:
            self.progress.setValue(value)
        if value == -1:
            val = self.progress.value() + 1
            self.progress.setValue(val)

    def abort_import_progress(self):
        self.progress.setLabelText(_('Cancelling ...'))
        self.progress.setMaximum(0)
        self.worker_object.abort()
        worker = Application.getWorker(self.worker_idx)
        worker.wait()
        self.progress.cancel()
        Application.destroyWorker(self.worker_idx)

    def finish_import_progress(self):
        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Import'), _('Dataset has been imported successfully'))
        self.progress.close()
        self.close()

    def error_import_progress(self, e):
        self.progress.cancel()
        mb = QtWidgets.QMessageBox()
        mb.warning(self, _('Import'), _('An error occured during import of dataset'))
    