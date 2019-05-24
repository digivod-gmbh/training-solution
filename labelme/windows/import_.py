from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets


class ImportWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super().__init__(parent)
        self.setWindowTitle(_('Import dataset'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowStaysOnTopHint)
    