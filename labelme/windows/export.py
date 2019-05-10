from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

class ExportWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):

        super(ExportWindow, self).__init__(parent)
        self.setWindowTitle('Export')

        formats_label = QtWidgets.QLabel('Format')

        formats = QtWidgets.QComboBox()
        formats.addItem('JSON')
        formats.addItem('REC')
        formats.addItem('TXT')
        formats.addItem('VOC')
        formats.addItem('COCO')

        format_group = QtWidgets.QGroupBox()
        format_group.setTitle('Format')
        format_group_layout = QtWidgets.QFormLayout()
        format_group.setLayout(format_group_layout)
        format_group_layout.addWidget(formats_label)
        format_group_layout.addWidget(formats)

        export_btn = QtWidgets.QPushButton('Export')
        cancel_btn = QtWidgets.QPushButton('Cancel')

        button_group = QtWidgets.QGroupBox()
        button_group_layout = QtWidgets.QHBoxLayout()
        button_group.setLayout(button_group_layout)
        button_group_layout.addWidget(export_btn)
        button_group_layout.addWidget(cancel_btn)
        button_group.setFixedHeight(50)

        scrollArea = QtWidgets.QScrollArea()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(format_group)
        layout.addWidget(button_group)
        scrollArea.setLayout(layout)
        
        self.setCentralWidget(scrollArea)

        self.resize(QtCore.QSize(400, 500))
        self.move(QtCore.QPoint(0, 0))
