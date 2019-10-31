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

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import deltree, WorkerDialog, replace_special_chars
from labelme.utils.map import Map
from labelme.utils import WorkerExecutor
from labelme.extensions.formats import *
from labelme.config import MessageType
from labelme.config.export import Export


class ExportWindow(WorkerDialog):

    def __init__(self, parent=None):
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
        self.data_folder.setReadOnly(True)

        data_folder_group = QtWidgets.QGroupBox()
        data_folder_group.setTitle(_('Data folder'))
        data_folder_group_layout = QtWidgets.QHBoxLayout()
        data_folder_group.setLayout(data_folder_group_layout)
        data_folder_group_layout.addWidget(self.data_folder)
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
        self.label_selection_scroll.setMinimumHeight(40)
        self.label_selection_scroll.setMaximumHeight(150)
        self.label_selection_scroll.setWidget(self.label_parent_widget)
        layout.addWidget(self.label_selection_scroll)

        labels = self.get_labels()
        if self.parent.lastOpenDir is not None and len(labels) > 0:
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
        validation_group.setTitle(_('Validation ratio'))
        validation_group_layout = QtWidgets.QGridLayout()
        validation_group.setLayout(validation_group_layout)
        validation_group_layout.addWidget(self.validation, 0, 0)
        validation_group_layout.addWidget(validation_label, 0, 1)
        layout.addWidget(validation_group)

        layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox()
        export_btn = button_box.addButton(_('Export'), QtWidgets.QDialogButtonBox.AcceptRole)
        export_btn.clicked.connect(self.export_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def get_labels(self):
        labels = []
        for i in range(len(self.parent.uniqLabelList)):
            labels.append(self.parent.uniqLabelList.item(i).text())
        logger.debug('Loaded labels: {}'.format(', '.join(labels)))
        return labels

    def setVisible(self, visible):
        if not self.has_opened_images():
            mb = QtWidgets.QMessageBox()
            mb.information(self, _('Export'), _('Please open a folder with images first'))
        else:
            super().setVisible(visible)

    def has_opened_images(self):
        image_list = self.parent.imageList
        labels = self.get_labels()
        if len(image_list) == 0 or len(labels) == 0:
            return False
        return True

    # def get_label_files_from_data_folder(self, data_folder, selected_labels = [], get_all_labels = False):
    #     label_files = []
    #     for root, dirs, files in os.walk(data_folder):
    #         for f in files:
    #             if LabelFile.is_label_file(f):
    #                 full_path = os.path.normpath(os.path.join(root, f))
    #                 try:
    #                     lf = LabelFile(full_path)
    #                 except:
    #                     continue
    #                 if get_all_labels:
    #                     label_files.append(full_path)
    #                     continue
    #                 labels = [s[0] for s in lf.shapes]
    #                 intersect_labels = set(labels) & set(selected_labels)
    #                 if len(intersect_labels) > 0:
    #                     label_files.append(full_path)
    #     num_label_files = len(label_files)
    #     logger.debug('Found {} label files in dataset folder "{}"'.format(num_label_files, data_folder))
    #     return label_files, num_label_files

    def cancel_btn_clicked(self):
        self.close()

    def load_labels(self, labels):
        image_list = self.parent.imageList
        if len(image_list) > 0:
            self.label_checkboxes = []
            labels.sort()
            for label in labels:
                checkbox = QtWidgets.QCheckBox(label)
                self.label_checkboxes.append(checkbox)
            self.label_selection_label.setVisible(True)
            self.label_selection_scroll.setVisible(True)
            self.update_label_checkboxes()

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

        try:
            import ptvsd
            ptvsd.debug_this_thread()
        except:
            pass
        
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

        dataset_name = replace_special_chars(self.data['dataset_name'])
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
