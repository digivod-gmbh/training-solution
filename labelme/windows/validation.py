import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from labelme.utils import WorkerDialog
from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Training
from labelme.extensions.networks import Network
from .inference import InferenceWindow
from labelme.config.export import Export


class ValidationWindow(WorkerDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Validation'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.input_image_file = QtWidgets.QLineEdit()
        input_image_file_browse_btn = QtWidgets.QPushButton(_('Browse'))
        input_image_file_browse_btn.clicked.connect(self.input_image_file_browse_btn_clicked)

        input_image_group = QtWidgets.QGroupBox()
        input_image_group.setTitle(_('Input image'))
        input_image_group_layout = QtWidgets.QGridLayout()
        input_image_group.setLayout(input_image_group_layout)
        input_image_group_layout.addWidget(self.input_image_file, 0, 0)
        input_image_group_layout.addWidget(input_image_file_browse_btn, 0, 1)
        layout.addWidget(input_image_group)

        self.training_folder = QtWidgets.QLineEdit()
        training_folder_browse_btn = QtWidgets.QPushButton(_('Browse'))
        training_folder_browse_btn.clicked.connect(self.training_folder_browse_btn_clicked)

        network_files_group = QtWidgets.QGroupBox()
        network_files_group.setTitle(_('Training'))
        network_files_group_layout = QtWidgets.QGridLayout()
        network_files_group.setLayout(network_files_group_layout)
        network_files_group_layout.addWidget(self.training_folder, 0, 0)
        network_files_group_layout.addWidget(training_folder_browse_btn, 0, 1)
        layout.addWidget(network_files_group)

        button_box = QtWidgets.QDialogButtonBox()
        validate_btn = button_box.addButton(_('Validate'), QtWidgets.QDialogButtonBox.AcceptRole)
        validate_btn.clicked.connect(self.validate_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def input_image_file_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('validation/last_input_image_dir', '')
        logger.debug('Restored value "{}" for setting validation/last_input_image_dir'.format(last_dir))
        filters = _('Image files') + ' (*.jpg *.jpeg *.png *.bmp)'
        image_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select input image'), last_dir, filters)
        if image_file:
            image_file = os.path.normpath(image_file)
            self.parent.settings.setValue('validation/last_input_image_dir', os.path.dirname(image_file))
            self.input_image_file.setText(image_file)

    def training_folder_browse_btn_clicked(self):
        project_folder = self.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))
        training_folder = os.path.join(project_folder, self.parent._config['project_training_folder'])
        training_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select training file'), training_folder)
        if training_folder:
            training_folder = os.path.normpath(training_folder)
            self.training_folder.setText(training_folder)

    def validate_btn_clicked(self):
        input_image_file = self.input_image_file.text()
        input_image_file_name = os.path.splitext(os.path.basename(input_image_file))[0]
        input_image_dir = os.path.dirname(input_image_file)
        if not input_image_file or not os.path.isfile(input_image_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Validation'), _('Please select a valid input image'))
            return

        training_folder = self.training_folder.text()
        if not training_folder or not os.path.isdir(training_folder):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Validation'), _('Please select a valid training folder'))
            return

        inferenceWin = InferenceWindow(self)
        inferenceWin.show()
        inferenceWin.start_inference(input_image_file, training_folder)
        self.close()

    def cancel_btn_clicked(self):
        self.close()
