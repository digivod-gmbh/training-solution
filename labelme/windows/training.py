from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

class TrainingWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):

        super(TrainingWindow, self).__init__(parent)
        self.setWindowTitle('Training')

        start_btn = QtWidgets.QPushButton('Start training')
        cancel_btn = QtWidgets.QPushButton('Cancel')

        button_group = QtWidgets.QGroupBox()
        button_group_layout = QtWidgets.QHBoxLayout()
        button_group.setLayout(button_group_layout)
        button_group_layout.addWidget(start_btn)
        button_group_layout.addWidget(cancel_btn)
        button_group.setFixedHeight(50)

        scrollArea = QtWidgets.QScrollArea()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(button_group)
        scrollArea.setLayout(layout)
        
        self.setCentralWidget(scrollArea)

        self.resize(QtCore.QSize(400, 500))
        self.move(QtCore.QPoint(0, 0))