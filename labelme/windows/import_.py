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
from labelme.config.export import Export


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
        self.formats.setCurrentIndex(0)
        self.formats.currentTextChanged.connect(self.on_format_change)
        self.selected_format = list(Export.config('formats').keys())[0]

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

    def on_format_change(self, value):
        formats = Export.config('formats')
        inv_formats = Export.invertDict(formats)
        if value in inv_formats:
            self.selected_format = inv_formats[value]
            logger.debug('Selected import format: {}'.format(self.selected_format))
        else:
            logger.debug('Import format not found: {}'.format(value))

    def data_browse_btn_clicked(self):
        ext_filter = False
        extension = Export.config('extensions')[self.selected_format]
        format_name = Export.config('formats')[self.selected_format]
        if extension != False:
            ext_filter = '{} {}({})'.format(format_name, _('files'), extension)
        last_dir = self.parent.settings.value('import/last_data_dir', '')
        logger.debug('Restored value "{}" for setting import/last_data_dir'.format(last_dir))
        if ext_filter:
            import_file_or_dir, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select dataset file'), last_dir, ext_filter)
        else:
            import_file_or_dir = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select dataset folder'), last_dir)
        if import_file_or_dir:
            import_file_or_dir = os.path.normpath(import_file_or_dir)
            self.parent.settings.setValue('import/last_data_dir', os.path.dirname(import_file_or_dir))
            self.data_folder.setText(import_file_or_dir)

    def output_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('import/last_output_dir', '')
        logger.debug('Restored value "{}" for setting import/last_output_dir'.format(last_dir))
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), last_dir)
        if output_folder:
            output_folder = os.path.normpath(output_folder)
            self.parent.settings.setValue('import/last_output_dir', output_folder)
            self.output_folder.setText(output_folder)

    def import_btn_clicked(self):
        data_folder_or_file = self.data_folder.text()
        if not data_folder_or_file or not (os.path.isdir(data_folder_or_file) or os.path.isfile(data_folder_or_file)):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('Please enter a valid dataset file or folder'))
            return

        output_folder = self.output_folder.text()
        if not output_folder or not os.path.isdir(output_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('Please enter a valid output folder'))
            return

        val = self.formats.currentText()
        formats = Export.config('formats')
        inv_formats = Export.invertDict(formats)
        if val not in inv_formats:
            logger.error('Import format {} could not be found'.format(val))
            return
        else:
            format_name = inv_formats[val]

        # Dataset
        dataset_format = Export.config('objects')[format_name]()
        if not dataset_format.isValidFormat(data_folder_or_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Import'), _('Invalid dataset format'))
            return

        dataset_format.setOutputFolder(output_folder)
        dataset_format.setInputFolderOrFile(data_folder_or_file)

        self.progress = QtWidgets.QProgressDialog(_('Initializing ...'), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.show()
        self.progress.setMaximum(100)
        self.progress.setLabelText(_('Initializing ...'))
        self.progress.setValue(0)

        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = ProgressObject(worker, dataset_format.importFolder, self.error_import_progress, dataset_format.abort, 
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
        
        # open import folder
        output_folder = os.path.normpath(self.output_folder.text())
        self.parent.importDirImages(output_folder)

        self.close()

    def error_import_progress(self, e):
        self.progress.cancel()
        mb = QtWidgets.QMessageBox()
        mb.warning(self, _('Import'), _('An error occured during import of dataset'))
    