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

import ptvsd
import traceback

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import Worker, ProgressObject, Application, AbortWorkerException, deltree
from labelme.utils.map import Map
from labelme.extensions.formats import *
from labelme.config import Export, MessageType


class ExportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None, labels=[]):
        self.parent = parent

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

    # def export_btn_clicked2(self):
    #     data_folder = self.data_folder.text()
    #     if not data_folder or not os.path.isdir(data_folder):
    #         mb = QtWidgets.QMessageBox
    #         mb.warning(self, _('Export'), _('Please enter a valid data folder'))
    #         return

    #     export_folder = os.path.normpath(self.export_folder.text())
    #     if not export_folder or not os.path.isdir(export_folder):
    #         mb = QtWidgets.QMessageBox
    #         mb.warning(self, _('Export'), _('Please enter a valid export folder'))
    #         return

    #     selected_labels = []
    #     for i, checkbox in enumerate(self.label_checkboxes):
    #         if checkbox.isChecked():
    #             selected_labels.append(checkbox.text())
    #     num_selected_labels = len(selected_labels)
    #     limit = Export.config('limits')['max_num_labels']
    #     if num_selected_labels > limit:
    #         mb = QtWidgets.QMessageBox
    #         mb.warning(self, _('Export'), _('Please select a maximum of {} labels').format(limit))
    #         return
    #     elif num_selected_labels <= 0:
    #         mb = QtWidgets.QMessageBox
    #         mb.warning(self, _('Export'), _('Please select at least 1 label'))
    #         return

    #     validation_ratio = int(self.validation.value()) / 100.0
    #     dataset_name = self.export_name.text()
    #     dataset_name = re.sub(r'[^a-zA-Z0-9 _-]+', '', dataset_name)

    #     if not dataset_name:
    #         mb = QtWidgets.QMessageBox
    #         mb.warning(self, _('Export'), _('Please enter a valid dataset name'))
    #         return
            
    #     export_dataset_folder = os.path.normpath(os.path.join(export_folder, dataset_name))
    #     if not os.path.isdir(export_dataset_folder):
    #         os.makedirs(export_dataset_folder)
    #     elif len(os.listdir(export_dataset_folder)) > 0:
    #         mb = QtWidgets.QMessageBox
    #         msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(export_dataset_folder)
    #         clicked_btn = mb.warning(self, _('Export'), msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    #         if clicked_btn != QtWidgets.QMessageBox.Yes:
    #             return
    #         else:
    #             deltree(export_dataset_folder)
    #             if not os.path.exists(export_dataset_folder):
    #                 os.makedirs(export_dataset_folder)

    #     if not os.path.isdir(export_dataset_folder):
    #         mb = QtWidgets.QMessageBox
    #         mb.warning(self, _('Export'), _('The selected output directory "{}" could not be created').format(export_dataset_folder))
    #         return

    #     self.progress = QtWidgets.QProgressDialog(_('Initializing ...'), _('Cancel'), 0, 100, self)
    #     self.set_default_window_flags(self.progress)
    #     self.progress.setWindowModality(Qt.ApplicationModal)
    #     self.progress.setMaximum(100)
    #     self.progress.setValue(0)
    #     self.progress.show()

    #     # Intermediate Format
    #     intermediate = IntermediateFormat()
    #     intermediate.setIncludedLabels(selected_labels)
    #     intermediate.setValidationRatio(validation_ratio)
    #     intermediate.addFromLabelFiles(data_folder, shuffle=False)

    #     self.progress.setMaximum(intermediate.getNumberOfSamples() + 5)
    #     self.progress.setLabelText(_('Loading data ...'))
    #     self.progress.setValue(0)

    #     val = self.formats.currentText()
    #     formats = Export.config('formats')
    #     inv_formats = Export.invertDict(formats)
    #     if val not in inv_formats:
    #         logger.error('Export format {} could not be found'.format(val))
    #         return
    #     else:
    #         format_name = inv_formats[val]

    #     args = Map({
    #         'validation_ratio': validation_ratio,
    #     })

    #     dataset_format = Export.config('objects')[format_name]()
    #     dataset_format.setIntermediateFormat(intermediate)
    #     dataset_format.setInputFolderOrFile(data_folder)
    #     dataset_format.setOutputFolder(export_dataset_folder)
    #     dataset_format.setArgs(args)

    #     worker_idx, worker = Application.createWorker()
    #     self.worker_idx = worker_idx
    #     self.worker_object = ProgressObject(worker, dataset_format.export, self.error_export_progress, dataset_format.abort, 
    #         self.update_export_progress, self.finish_export_progress)
    #     dataset_format.setThread(self.worker_object)

    #     self.progress.canceled.disconnect()
    #     self.progress.canceled.connect(self.abort_export_progress)
    #     worker.addObject(self.worker_object)
    #     worker.start()

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

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

    # def update_export_progress(self, msg=None, value=None):
    #     if self.progress.wasCanceled():
    #         return
    #     if msg:
    #         self.progress.setLabelText(msg)
    #     if value is not None:
    #         self.progress.setValue(value)
    #     if value == -1:
    #         val = self.progress.value() + 1
    #         self.progress.setValue(val)

    # def abort_export_progress(self):
    #     self.progress.setLabelText(_('Cancelling ...'))
    #     self.progress.setMaximum(0)
    #     self.worker_object.abort()
    #     worker = Application.getWorker(self.worker_idx)
    #     worker.wait()
    #     self.progress.cancel()
    #     Application.destroyWorker(self.worker_idx)

    # def finish_export_progress(self):
    #     mb = QtWidgets.QMessageBox()
    #     mb.information(self, _('Export'), _('Dataset has been exported successfully'))
    #     self.progress.close()
    #     self.close()

    # def error_export_progress(self, e):
    #     self.progress.cancel()
    #     mb = QtWidgets.QMessageBox()
    #     mb.warning(self, _('Export'), _('An error occured during export of dataset'))

    def export_btn_clicked(self):
        # Data
        data = {
            'data_folder': os.path.normpath(self.data_folder.text()),
            'export_folder': os.path.normpath(self.export_folder.text()),
            'selected_labels': [x.text() for x in self.label_checkboxes if x.isChecked()],
            'validation_ratio': int(self.validation.value()) / 100.0,
            'dataset_name': re.sub(r'[^a-zA-Z0-9 _-]+', '', self.export_name.text()),
            'max_num_labels': Export.config('limits')['max_num_labels'],
            'selected_format': self.formats.currentText(),
        }
        data['export_dataset_folder'] = os.path.normpath(os.path.join(data['export_folder'], data['dataset_name']))

        self.progress = QtWidgets.QProgressDialog(_('Initializing ...'), _('Cancel'), 0, 100, self)
        self.set_default_window_flags(self.progress)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.show()

        # Thread
        self.worker = ExportWorker(data)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        #self.worker.finished.connect(self.thread.quit)
        self.worker.terminate.connect(self.terminate)
        self.worker.progress.connect(self.update_progress)
        self.worker.message.connect(self.show_message)
        self.worker.confirm.connect(self.show_confirm)
        self.thread.started.connect(self.worker.prepare_export) # function to start with
        #self.thread.finished.connect(self.finished)

        self.thread.finished.connect(self.finish_thread)

        #self.progress.canceled.disconnect()
        self.progress.canceled.connect(self.abort_progress)
        self.thread.start()

    def terminate(self):
        self.thread.quit()

    def finish_thread(self):
        logger.debug("foo")

    def update_progress(self, msg=None, value=None, maximum=None):
        if self.progress.wasCanceled():
            return
        if msg:
            self.progress.setLabelText(msg)
        if value is not None:
            self.progress.setValue(value)
        if value == -1:
            val = self.progress.value() + 1
            self.progress.setValue(val)
        if maximum is not None:
            self.progress.setMaximum(maximum)
            
    def abort_progress(self):
        self.progress.setLabelText(_('Cancelling ...'))
        self.progress.setMaximum(0)
        self.worker.cancel()
        self.thread.wait()
        self.progress.cancel()

    def show_message(self, title, msg, kind=None):
        mb = QtWidgets.QMessageBox
        if kind == MessageType.Warning:
            mb.warning(self, title, msg)
        elif kind == MessageType.Error:
            mb.critical(self, title, msg)
        elif kind == MessageType.Question:
            mb.question(self, title, msg)
        else:
            mb.information(self, title, msg)

    def show_confirm(self, title, msg, kind=None):
        clicked_btn = False
        mb = QtWidgets.QMessageBox
        if kind == MessageType.Warning:
            clicked_btn = mb.warning(self, title, msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        elif kind == MessageType.Error:
            clicked_btn = mb.critical(self, title, msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        elif kind == MessageType.Question:
            clicked_btn = mb.question(self, title, msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        else:
            clicked_btn = mb.information(self, title, msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        result = clicked_btn == QtWidgets.QMessageBox.Yes
        self.worker.confirm_result(result)

    def finished(self):
        #self.progress.close()
        #self.close()
        pass
 

class ExportWorker(QObject):

    finished = Signal()
    terminate = Signal()
    progress = Signal(str, int, int)
    message = Signal(str, str, int)
    confirm = Signal(str, str, int)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.workers = []
        self.isAborted = False

    def cancel(self):
        for w in self.workers:
            w.abort()

    # Sub-Worker callbacks

    @Slot()
    def on_finish(self):
        self.message.emit(_('Export'), _('Dataset has been exported successfully'), MessageType.Info)
        self.finished.emit()

    @Slot()
    def on_error(self):
        self.message.emit(_('Export'), _('An error occured during export of dataset'), MessageType.Error)
        self.finished.emit()
        
    @Slot(str, int)
    def on_update(self, msg, value=None):
        self.progress.emit(msg, value, None)

    def update(self, msg, value=None):
        self.progress.emit(msg, value, None)

    # Confirm

    def do_confirm(self, title, msg, kind):
        self.wait_for_confirm = True
        self.confirm_value = None
        self.confirm.emit(title, msg, kind)
        while self.wait_for_confirm:
            time.sleep(0.5)
        return self.confirm_value

    @Slot(bool)
    def confirm_result(self, result):
        self.wait_for_confirm = False
        self.confirm_value = result

    # Error

    def do_error(self, ex):
        logger.error(traceback.format_exc())
        self.message.emit(_('Export'), str(ex), MessageType.Error)
        self.finished.emit()

    # Preparing

    @Slot()
    def prepare_export(self):
        try:
            logger.debug("Prepare export")
            ptvsd.debug_this_thread()
            
            data_folder = self.data['data_folder']
            if not data_folder or not os.path.isdir(data_folder):
                self.message.emit(_('Export'), _('Please enter a valid data folder'), MessageType.Warning)
                self.finished.emit()
                return

            export_folder = self.data['export_folder']
            if not export_folder or not os.path.isdir(export_folder):
                self.message.emit(_('Export'), _('Please enter a valid export folder'), MessageType.Warning)
                self.finished.emit()
                return

            selected_labels = self.data['selected_labels']
            num_selected_labels = len(selected_labels)
            limit = self.data['max_num_labels']
            if num_selected_labels > limit:
                self.message.emit(_('Export'), _('Please select a maximum of {} labels').format(limit), MessageType.Warning)
                self.finished.emit()
                return
            elif num_selected_labels <= 0:
                self.message.emit(_('Export'), _('Please select at least 1 label'), MessageType.Warning)
                self.finished.emit()
                return

            dataset_name = self.data['dataset_name']
            if not dataset_name:
                self.message.emit(_('Export'), _('Please enter a valid dataset name'), MessageType.Warning)
                self.finished.emit()
                return

            export_dataset_folder = self.data['export_dataset_folder']
            if not os.path.isdir(export_dataset_folder):
                os.makedirs(export_dataset_folder)
            elif len(os.listdir(export_dataset_folder)) > 0:
                msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(export_dataset_folder)
                if self.do_confirm(_('Export'), msg, MessageType.Warning):
                    deltree(export_dataset_folder)
                    time.sleep(0.5) # wait for deletion to be finished
                    if not os.path.exists(export_dataset_folder):
                        os.makedirs(export_dataset_folder)
                else:
                    self.finished.emit()
                    return

            if not os.path.isdir(export_dataset_folder):
                self.message.emit(_('Export'), _('The selected output directory "{}" could not be created').format(export_dataset_folder), MessageType.Warning)
                self.finished.emit()
                return

            selected_format = self.data['selected_format']
            all_formats = Export.config('formats')
            inv_formats = Export.invertDict(all_formats)
            if selected_format not in inv_formats:
                logger.error('Export format {} could not be found'.format(selected_format))
                self.finished.emit()
                return
            else:
                self.data['format_name'] = inv_formats[selected_format]

            self.start_export()

        except Exception as e:
            logger.error(traceback.format_exc())
            self.do_error(e)

    # Export

    def start_export(self):
        try:
            logger.debug("Start export")

            selected_labels = self.data['selected_labels']
            validation_ratio = self.data['validation_ratio']
            data_folder = self.data['data_folder']
            export_dataset_folder = self.data['export_dataset_folder']
            format_name = self.data['format_name']

            intermediate = IntermediateFormat()
            self.workers.append(intermediate)
            intermediate.setIncludedLabels(selected_labels)
            intermediate.setValidationRatio(validation_ratio)
            intermediate.addFromLabelFiles(data_folder, shuffle=False)

            self.progress.emit(_('Loading data ...'), 0, intermediate.getNumberOfSamples() + 5)

            args = Map({
                'validation_ratio': validation_ratio,
            })

            dataset_format = Export.config('objects')[format_name](self)
            self.workers.append(dataset_format)
            dataset_format.setIntermediateFormat(intermediate)
            dataset_format.setInputFolderOrFile(data_folder)
            dataset_format.setOutputFolder(export_dataset_folder)
            dataset_format.setArgs(args)

            if self.isAborted:
                return

            dataset_format.export()

            self.finished.emit()

        except AbortWorkerException as e:
            logger.debug('Worker thread aborted')
            self.terminate.emit()

        except Exception as e:
            logger.error(traceback.format_exc())
            self.do_error(e)
        
