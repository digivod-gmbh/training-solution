from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

import importlib
import os

class TrainingWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):

        self.config = {
            'networks': {
                _('YoloV3'): 'yolov3'
            }
        }
        self.parent = parent

        super(TrainingWindow, self).__init__(parent)
        self.setWindowTitle(_('Training'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.networks = QtWidgets.QComboBox()
        for key in self.config['networks']:
            self.networks.addItem(key)

        network_group = QtWidgets.QGroupBox()
        network_group.setTitle(_('Network'))
        network_group_layout = QtWidgets.QVBoxLayout()
        network_group.setLayout(network_group_layout)
        network_group_layout.addWidget(self.networks)
        layout.addWidget(network_group)

        button_box = QtWidgets.QDialogButtonBox()
        training_btn = button_box.addButton(_('Start Training'), QtWidgets.QDialogButtonBox.AcceptRole)
        training_btn.clicked.connect(self.training_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def training_btn_clicked(self):

        numTasks = 10000000
        progress = QtWidgets.QProgressDialog(_('Training ...'), _('Cancel'), 0, numTasks, self)
        self.set_default_window_flags(progress)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.show()

        key = self.networks.currentText()
        if key in self.config['networks']:
            func_name = self.config['networks'][key]
            training_func = getattr(self, func_name)
            training_func()

    def cancel_btn_clicked(self):
        self.close()

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)