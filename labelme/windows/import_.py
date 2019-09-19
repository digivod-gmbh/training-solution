from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import os

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import deltree, WorkerDialog
from labelme.extensions.thread import WorkerExecutor
from labelme.config import MessageType
from labelme.utils.map import Map
from labelme.extensions.formats import *
from labelme.config.export import Export


class ImportWindow(WorkerDialog):

    def __init__(self, parent=None):
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
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        dataset_folder = os.path.join(project_folder, self.parent._config['project_dataset_folder'])
        if ext_filter:
            import_file_or_dir, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select dataset file'), dataset_folder, ext_filter)
        else:
            import_file_or_dir = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select dataset folder'), dataset_folder)
        if import_file_or_dir:
            import_file_or_dir = os.path.normpath(import_file_or_dir)
            self.parent.settings.setValue('import/last_data_dir', os.path.dirname(import_file_or_dir))
            self.data_folder.setText(import_file_or_dir)

    def output_browse_btn_clicked(self):
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        import_folder = os.path.join(project_folder, self.parent._config['project_import_folder'])
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), import_folder)
        if output_folder:
            output_folder = os.path.normpath(output_folder)
            self.parent.settings.setValue('import/last_output_dir', output_folder)
            self.output_folder.setText(output_folder)

    def cancel_btn_clicked(self):
        self.close()

    def import_btn_clicked(self):
        # Data
        data = {
            'data_folder': self.data_folder.text(),
            'output_folder': self.output_folder.text(),
            'selected_format': self.formats.currentText(),
        }

        # Execution
        executor = ImportExecutor(data)
        self.run_thread(executor, self.finish_import)

    def finish_import(self):
        # Open import folder
        output_folder = os.path.normpath(self.output_folder.text())
        self.parent.importDirImages(output_folder)

        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Import'), _('Dataset has been imported successfully'))

        self.close()


class ImportExecutor(WorkerExecutor):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        logger.debug('Prepare import')
        
        try:
            import ptvsd
            ptvsd.debug_this_thread()
        except:
            pass

        data_folder_or_file = self.data['data_folder']
        is_data_folder_valid = True
        if not data_folder_or_file:
            is_data_folder_valid = False
        data_folder_or_file = os.path.normpath(data_folder_or_file)
        if not (os.path.isdir(data_folder_or_file) or os.path.isfile(data_folder_or_file)):
            is_data_folder_valid = False
        if not is_data_folder_valid:
            self.thread.message.emit(_('Import'), _('Please enter a valid dataset file or folder'), MessageType.Warning)
            self.abort()
            return

        output_folder = self.data['output_folder']
        is_output_folder_valid = True
        if not output_folder:
            is_output_folder_valid = False
        output_folder = os.path.normpath(output_folder)
        if not os.path.isdir(output_folder):
            is_output_folder_valid = False
        if not is_output_folder_valid:
            self.thread.message.emit(_('Import'), _('Please enter a valid output folder'), MessageType.Warning)
            self.abort()
            return

        selected_format = self.data['selected_format']
        all_formats = Export.config('formats')
        inv_formats = Export.invertDict(all_formats)
        if selected_format not in inv_formats:
            self.thread.message.emit(_('Import'), _('Import format {} could not be found').format(selected_format), MessageType.Warning)
            self.abort()
            return
        else:
            self.data['format_name'] = inv_formats[selected_format]
        format_name = self.data['format_name']

        # Dataset
        dataset_format = Export.config('objects')[format_name]()
        if not dataset_format.isValidFormat(data_folder_or_file):
            self.thread.message.emit(_('Import'), _('Invalid dataset format'), MessageType.Warning)
            self.abort()
            return

        dataset_format.setAbortable(self.abortable)
        dataset_format.setThread(self.thread)
        dataset_format.setOutputFolder(output_folder)
        dataset_format.setInputFolderOrFile(data_folder_or_file)

        self.checkAborted()

        dataset_format.importFolder()
