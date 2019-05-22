import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Export, Training
from labelme.extensions.networks import Network


class ValidationWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

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

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)

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
        last_dir = self.parent.settings.value('validation/last_training_dir', '')
        logger.debug('Restored value "{}" for setting validation/last_training_dir'.format(last_dir))
        training_folder = QtWidgets.QFileDialog.getExistingDirectory(self, _('Select training file'), last_dir)
        if training_folder:
            training_folder = os.path.normpath(training_folder)
            self.parent.settings.setValue('validation/last_training_dir', training_folder)
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

        config_file = os.path.join(training_folder, Training.config('config_file'))

        network = Network()
        network_config = network.loadConfig(config_file)

        architecture_file = ''
        weights_file = ''
        files = network_config.files
        for f in files:
            if '.json' in f:
                architecture_file = os.path.join(training_folder, f)
            elif '.params' in f:
                weights_file = os.path.join(training_folder, f)

        dataset_folder = network_config.dataset
        label_file = os.path.join(dataset_folder, Export.config('labels_file'))

        network.inference(input_image_file, label_file, architecture_file, weights_file, args = None)

        # # Load training data
        # training = Training.read_training_config(training_folder)
        # architecture_file = os.path.join(training_folder, training.architecture)
        # weights_file = os.path.join(training_folder, training.weights)
        # label_list_file = os.path.normpath(os.path.join(training_folder, training.label_list))
        # args = Map(training.args)

        # # Load network
        # net_objects = Training.config('objects')
        # network = net_objects[training.network]()
        # network.inference(input_image_file, label_list_file, architecture_file, weights_file, args)

    def cancel_btn_clicked(self):
        self.close()

