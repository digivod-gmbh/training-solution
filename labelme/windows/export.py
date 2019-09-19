from qtpy import QtCore
from qtpy.QtCore import Qt, Signal, Slot, QThread, QObject
from qtpy import QtGui
from qtpy import QtWidgets

import os
import re
import sys
import json
import time
import shutil
import importlib
import subprocess

import traceback
import ptvsd

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import deltree, WorkerDialog
from labelme.utils.map import Map
from labelme.extensions.thread import WorkerExecutor
from labelme.extensions.formats import *
from labelme.config import MessageType
from labelme.config.export import Export


class ExportWindow(WorkerDialog):

    def __init__(self, parent=None, labels=[]):
        super().__init__(parent)
        self.setWindowTitle(_('Export dataset'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        logger.debug('Open export window with labels: {}'.format(', '.join(labels)))

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
        #data_browse_btn = QtWidgets.QPushButton(_('Browse'))
        #data_browse_btn.clicked.connect(self.data_browse_btn_clicked)

        data_folder_group = QtWidgets.QGroupBox()
        data_folder_group.setTitle(_('Data folder'))
        data_folder_group_layout = QtWidgets.QHBoxLayout()
        data_folder_group.setLayout(data_folder_group_layout)
        data_folder_group_layout.addWidget(self.data_folder)
        #data_folder_group_layout.addWidget(data_browse_btn)
        layout.addWidget(data_folder_group)

        self.label_checkboxes = []
        self.label_selection_label = QtWidgets.QLabel(_('Label selection'))
        self.label_selection_label.setVisible(False)
        layout.addWidget(self.label_selection_label)

        self.label_parent_widget = QtWidgets.QWidget()
        self.label_parent_widget.setLayout(QtWidgets.QVBoxLayout())

        self.label_selection_scroll = QtWidgets.QScrollArea()
        self.label_selection_scroll.setVisible(False)
        self.label_selection_scroll.setWidgetResizable(True)
        self.label_selection_scroll.setFixedHeight(100)
        self.label_selection_scroll.setWidget(self.label_parent_widget)
        layout.addWidget(self.label_selection_scroll)

        if self.parent.lastOpenDir is not None:
            self.data_folder.setText(self.parent.lastOpenDir)
            self.load_labels(labels)

        self.export_folder = QtWidgets.QLineEdit()
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        self.export_folder.setText(os.path.join(project_folder, self.parent._config['project_dataset_folder']))
        self.export_folder.setReadOnly(True)
        export_browse_btn = QtWidgets.QPushButton(_('Browse'))
        export_browse_btn.clicked.connect(self.export_browse_btn_clicked)

        export_name_label = QtWidgets.QLabel(_('Dataset name'))
        self.export_name = QtWidgets.QLineEdit()

        export_folder_group = QtWidgets.QGroupBox()
        export_folder_group.setTitle(_('Export folder'))
        export_folder_group_layout = QtWidgets.QGridLayout()
        export_folder_group.setLayout(export_folder_group_layout)
        export_folder_group_layout.addWidget(self.export_folder, 0, 0, 1, 2)
        export_folder_group_layout.addWidget(export_browse_btn, 0, 2)
        export_folder_group_layout.addWidget(export_name_label, 1, 0, 1, 3)
        export_folder_group_layout.addWidget(self.export_name, 2, 0, 1, 3)
        layout.addWidget(export_folder_group)

        self.validation = QtWidgets.QSpinBox()
        self.validation.setValue(0)
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

    def get_label_files_from_data_folder(self, data_folder, selected_labels = [], get_all_labels = False):
        label_files = []
        for root, dirs, files in os.walk(data_folder):
            for f in files:
                if LabelFile.is_label_file(f):
                    full_path = os.path.normpath(os.path.join(data_folder, f))
                    try:
                        lf = LabelFile(full_path)
                    except:
                        continue
                    if get_all_labels:
                        label_files.append(full_path)
                        continue
                    labels = [s[0] for s in lf.shapes]
                    intersect_labels = set(labels) & set(selected_labels)
                    if len(intersect_labels) > 0:
                        label_files.append(full_path)
        num_label_files = len(label_files)
        logger.debug('Found {} label files in dataset folder "{}"'.format(num_label_files, data_folder))
        return label_files, num_label_files

    def cancel_btn_clicked(self):
        self.close()

    # def data_browse_btn_clicked(self):
    #     last_dir = self.parent.settings.value('export/last_dataset_dir', '')
    #     logger.debug('Restored value "{}" for setting export/last_dataset_dir'.format(last_dir))
    #     data_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select data folder'), last_dir)
    #     if data_folder:
    #         data_folder = os.path.normpath(data_folder)
    #         self.parent.settings.setValue('export/last_dataset_dir', data_folder)
    #         self.data_folder.setText(data_folder)
    #         self.load_labels_from_data_folder(data_folder)

    def load_labels(self, labels):
        self.label_checkboxes = []
        labels.sort()
        for label in labels:
            checkbox = QtWidgets.QCheckBox(label)
            self.label_checkboxes.append(checkbox)
        self.update_label_checkboxes()

    # def load_labels_from_data_folder(self, data_folder):
    #     self.progress = QtWidgets.QProgressDialog(_('Loading dataset ...'), _('Cancel'), 0, 100, self)
    #     self.set_default_window_flags(self.progress)
    #     self.progress.setWindowModality(Qt.ApplicationModal)
    #     self.progress.show()
    #     self.progress.setMaximum(100)
    #     self.progress.setValue(0)
    #     label_files, num_label_files = self.get_label_files_from_data_folder(data_folder, get_all_labels=True)
    #     self.progress.setMaximum(num_label_files + 1)
    #     self.progress.setValue(1)
    #     self.label_checkboxes = []
    #     label_set = set()
    #     for i, label_file in enumerate(label_files):
    #         lf = LabelFile(label_file)
    #         labels = [s[0] for s in lf.shapes]
    #         label_set.update(labels)
    #         self.progress.setValue(i + 2)
    #         if self.progress.wasCanceled():
    #             break
    #     if self.progress.wasCanceled():
    #         self.data_folder.setText('')
    #         return
    #     self.label_selection_label.setVisible(True)
    #     self.label_selection_scroll.setVisible(True)
    #     label_list = list(label_set)
    #     label_list.sort()
    #     logger.debug('Found labels {} in folder {}'.format(label_list, data_folder))
    #     for label in label_list:
    #         checkbox = QtWidgets.QCheckBox(label)
    #         self.label_checkboxes.append(checkbox)
    #     self.update_label_checkboxes()
    #     self.progress.close()

    def update_label_checkboxes(self):
        while self.label_parent_widget.layout().count():
            child = self.label_parent_widget.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        max_num_labels = Export.config('limits')['max_num_labels']
        for i, checkbox in enumerate(self.label_checkboxes):
            checkbox.setChecked(i < max_num_labels)
            self.label_parent_widget.layout().addWidget(checkbox)

    def export_browse_btn_clicked(self):
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        export_folder = os.path.join(project_folder, self.parent._config['project_dataset_folder'])
        export_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select output folder'), export_folder)
        if export_folder:
            export_folder = os.path.normpath(export_folder)
            self.export_folder.setText(export_folder)

    def export_btn_clicked(self):
        # Data
        data = {
            'data_folder': self.data_folder.text(),
            'export_folder': self.export_folder.text(),
            'selected_labels': [x.text() for x in self.label_checkboxes if x.isChecked()],
            'validation_ratio': int(self.validation.value()) / 100.0,
            'dataset_name': re.sub(r'[^a-zA-Z0-9 _-]+', '', self.export_name.text()),
            'max_num_labels': Export.config('limits')['max_num_labels'],
            'selected_format': self.formats.currentText(),
        }

        # Execution
        executor = ExportExecutor(data)
        self.run_thread(executor, self.finish_export)

    def finish_export(self):
        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Export'), _('Dataset has been exported successfully'))
        self.close()


class ExportExecutor(WorkerExecutor):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        logger.debug('Prepare export')
        ptvsd.debug_this_thread()
        
        data_folder = self.data['data_folder']
        is_data_folder_valid = True
        if not data_folder:
            is_data_folder_valid = False
        data_folder = os.path.normpath(data_folder)
        if not os.path.isdir(data_folder):
            is_data_folder_valid = False
        if not is_data_folder_valid:
            self.thread.message.emit(_('Export'), _('Please enter a valid data folder'), MessageType.Warning)
            self.abort()
            return

        export_folder = self.data['export_folder']
        is_export_folder_valid = True
        if not export_folder:
            is_export_folder_valid = False
        export_folder = os.path.normpath(export_folder)
        if not os.path.isdir(export_folder):
            is_export_folder_valid = False
        if not is_export_folder_valid:
            self.thread.message.emit(_('Export'), _('Please enter a valid export folder'), MessageType.Warning)
            self.abort()
            return

        selected_labels = self.data['selected_labels']
        num_selected_labels = len(selected_labels)
        limit = self.data['max_num_labels']
        if num_selected_labels > limit:
            self.thread.message.emit(_('Export'), _('Please select a maximum of {} labels').format(limit), MessageType.Warning)
            self.abort()
            return
        elif num_selected_labels <= 0:
            self.thread.message.emit(_('Export'), _('Please select at least 1 label'), MessageType.Warning)
            self.abort()
            return

        dataset_name = self.data['dataset_name']
        if not dataset_name:
            self.thread.message.emit(_('Export'), _('Please enter a valid dataset name'), MessageType.Warning)
            self.abort()
            return

        export_dataset_folder = os.path.normpath(os.path.join(self.data['export_folder'], self.data['dataset_name']))
        if not os.path.isdir(export_dataset_folder):
            os.makedirs(export_dataset_folder)
        elif len(os.listdir(export_dataset_folder)) > 0:
            msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(export_dataset_folder)
            if self.doConfirm(_('Export'), msg, MessageType.Warning):
                deltree(export_dataset_folder)
                time.sleep(0.5) # wait for deletion to be finished
                if not os.path.exists(export_dataset_folder):
                    os.makedirs(export_dataset_folder)
            else:
                self.abort()
                return

        if not os.path.isdir(export_dataset_folder):
            self.thread.message.emit(_('Export'), _('The selected output directory "{}" could not be created').format(export_dataset_folder), MessageType.Warning)
            self.abort()
            return

        selected_format = self.data['selected_format']
        all_formats = Export.config('formats')
        inv_formats = Export.invertDict(all_formats)
        if selected_format not in inv_formats:
            self.thread.message.emit(_('Export'), _('Export format {} could not be found').format(selected_format), MessageType.Warning)
            self.abort()
            return
        else:
            self.data['format_name'] = inv_formats[selected_format]

        logger.debug('Start export')

        selected_labels = self.data['selected_labels']
        validation_ratio = self.data['validation_ratio']
        data_folder = self.data['data_folder']
        export_dataset_folder = self.data['export_dataset_folder']
        format_name = self.data['format_name']

        self.checkAborted()

        intermediate = IntermediateFormat()
        intermediate.setAbortable(self.abortable)
        intermediate.setThread(self.thread)
        intermediate.setIncludedLabels(selected_labels)
        intermediate.setValidationRatio(validation_ratio)
        intermediate.addFromLabelFiles(data_folder, shuffle=False)

        self.thread.update.emit(_('Loading data ...'), 0, intermediate.getNumberOfSamples() + 5)

        args = Map({
            'validation_ratio': validation_ratio,
        })

        dataset_format = Export.config('objects')[format_name]()
        dataset_format.setAbortable(self.abortable)
        dataset_format.setThread(self.thread)
        dataset_format.setIntermediateFormat(intermediate)
        dataset_format.setInputFolderOrFile(data_folder)
        dataset_format.setOutputFolder(export_dataset_folder)
        dataset_format.setArgs(args)

        self.checkAborted()

        dataset_format.export()

