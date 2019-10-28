from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import os

from labelme.logger import logger
from labelme.config.export import Export
from labelme.widgets import HelpLabel, HelpCheckbox, HelpGroupBox


class SettingsWindow(QtWidgets.QDialog):

    def __init__(self, parent=None, prevent_close=False):
        self.parent = parent

        super().__init__(parent)
        self.setWindowTitle(_('Settings'))
        self.set_default_window_flags(self, prevent_close)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.project_folder = QtWidgets.QLineEdit()
        project_browse_btn = QtWidgets.QPushButton(_('Browse'))
        project_browse_btn.clicked.connect(self.project_browse_btn_clicked)

        saved_project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(saved_project_folder))
        self.project_folder.setText(saved_project_folder)
        self.project_folder.setReadOnly(True)
        self.project_folder.setFixedWidth(250)

        project_folder_group = HelpGroupBox('Settings_ProjectFolder', _('Project folder'))
        project_folder_group_layout = QtWidgets.QHBoxLayout()
        project_folder_group.widget.setLayout(project_folder_group_layout)
        project_folder_group_layout.addWidget(self.project_folder)
        project_folder_group_layout.addWidget(project_browse_btn)
        layout.addWidget(project_folder_group)

        saved_single_label_mode = self.parent.settings.value('settings/project/single_label_mode', '0')
        logger.debug('Restored value "{}" for setting settings/project/single_label_mode'.format(saved_single_label_mode))
        self.single_label_mode_checkbox = HelpCheckbox('Settings_SingleLabelMode', _('Single-label mode'))

        saved_single_label_name = self.parent.settings.value('settings/project/single_label_name', '')
        logger.debug('Restored value "{}" for setting settings/project/single_label_name'.format(saved_single_label_name))
        self.single_label_name_label = QtWidgets.QLabel(_('Label name'))
        self.single_label_name = QtWidgets.QLineEdit(saved_single_label_name)
        if saved_single_label_mode == '1':
            self.single_label_mode_checkbox.widget.setCheckState(Qt.Checked)
        else:
            self.single_label_mode_checkbox.widget.setCheckState(Qt.Unchecked)
            self.single_label_name_label.hide()
            self.single_label_name.hide()
        self.single_label_mode_checkbox.widget.stateChanged.connect(self.on_change_single_label_mode)

        single_label_group = QtWidgets.QGroupBox()
        single_label_group.setTitle(_('Labels'))
        single_label_group_layout = QtWidgets.QVBoxLayout()
        single_label_group.setLayout(single_label_group_layout)
        single_label_group_layout.addWidget(self.single_label_mode_checkbox)
        single_label_group_layout.addWidget(self.single_label_name_label)
        single_label_group_layout.addWidget(self.single_label_name)
        layout.addWidget(single_label_group)

        layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox()
        save_btn = button_box.addButton(_('Save'), QtWidgets.QDialogButtonBox.AcceptRole)
        save_btn.clicked.connect(self.save_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def project_browse_btn_clicked(self):
        project_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select project folder'))
        if project_folder:
            project_folder = os.path.normpath(project_folder)
            self.project_folder.setText(project_folder)

    def on_change_single_label_mode(self):
        if self.single_label_mode_checkbox.widget.isChecked():
            self.single_label_name_label.show()
            self.single_label_name.show()
        else:
            self.single_label_name_label.hide()
            self.single_label_name.hide()

    def save_btn_clicked(self):
        # Project folder
        project_folder = self.project_folder.text()
        if not self.check_folder(project_folder):
            return
        project_folder = os.path.normpath(project_folder)
        self.parent.settings.setValue('settings/project/folder', project_folder)
        project_directories = [
            self.parent._config['project_dataset_folder'],
            self.parent._config['project_training_folder'],
            self.parent._config['project_import_folder'],
        ]
        for directory in project_directories:
            abs_directory = os.path.join(project_folder, directory)
            if not os.path.isdir(abs_directory):
                os.makedirs(abs_directory)

        # Single label mode
        single_label_mode = self.single_label_mode_checkbox.widget.isChecked()
        label_name = self.single_label_name.text()
        if single_label_mode:
            if len(label_name) <= 0:
                msg = _('You must specify a label name in single-label mode')
                QtWidgets.QMessageBox.warning(self, _('Settings'), msg)
                return
        else:
            label_name = ''
        self.parent.settings.setValue('settings/project/single_label_mode', '1' if single_label_mode else '0')
        self.parent.settings.setValue('settings/project/single_label_name', label_name)

        self.close()

    def cancel_btn_clicked(self):
        project_folder = self.project_folder.text()
        if not self.check_folder(project_folder):
            return
        self.close()

    def check_folder(self, project_folder):
        if not project_folder:
            msg = _('Project folder must not be empty')
            QtWidgets.QMessageBox.warning(self, _('Settings'), msg)
            return False
        return True

    def set_default_window_flags(self, obj, prevent_close):
        if prevent_close:
            obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint)
        else:
            obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

