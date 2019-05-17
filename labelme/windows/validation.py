import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from labelme.windows import Training
from labelme.utils.map import Map


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

        self.training_file = QtWidgets.QLineEdit()
        training_file_browse_btn = QtWidgets.QPushButton(_('Browse'))
        training_file_browse_btn.clicked.connect(self.training_file_browse_btn_clicked)

        network_files_group = QtWidgets.QGroupBox()
        network_files_group.setTitle(_('Training'))
        network_files_group_layout = QtWidgets.QGridLayout()
        network_files_group.setLayout(network_files_group_layout)
        network_files_group_layout.addWidget(self.training_file, 0, 0)
        network_files_group_layout.addWidget(training_file_browse_btn, 0, 1)
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
        filters = _('Image files') + ' (*.jpg *.jpeg *.png *.bmp)'
        image_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select input image'), '', filters)
        if image_file:
            image_file = os.path.normpath(image_file)
            self.input_image_file.setText(image_file)

    def training_file_browse_btn_clicked(self):
        filters = _('Training file') + ' (*{})'.format(Training.config('config_file_extension'))
        training_file, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, _('Select training file'), '', filters)
        if training_file:
            training_file = os.path.normpath(training_file)
            self.training_file.setText(training_file)

    def validate_btn_clicked(self):
        input_image_file = self.input_image_file.text()
        input_image_file_name = os.path.splitext(os.path.basename(input_image_file))[0]
        input_image_dir = os.path.dirname(input_image_file)
        if not input_image_file or not os.path.isfile(input_image_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Validation'), _('Please select a valid input image'))
            return

        training_file = self.training_file.text()
        training_file_name = os.path.splitext(os.path.basename(training_file))[0]
        training_dir = os.path.dirname(training_file)
        if not training_file or not os.path.isfile(training_file):
            mb = QtWidgets.QMessageBox
            mb.warning(self, _('Validation'), _('Please select a valid training file'))
            return

        # Load training data
        training = Training.read_training_config(training_file)
        architecture_file = os.path.join(training_dir, training.architecture)
        weights_file = os.path.join(training_dir, training.weights)
        label_list_file = os.path.normpath(os.path.join(training_dir, training.label_list))
        args = Map(training.args)

        # Load network
        net_objects = Training.config('objects')
        network = net_objects[training.network]()
        network.inference(input_image_file, label_list_file, architecture_file, weights_file, args)

    def cancel_btn_clicked(self):
        self.close()
