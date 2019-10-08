import time
import os

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.extensions.thread import WorkerExecutor
from labelme.utils import WorkerDialog
from labelme.label_file import LabelFile
from labelme.shape import Shape


class ImageImportWindow(WorkerDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Import'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        status_label = QtWidgets.QLabel(_('Importing ...'))
        layout.addWidget(status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox()
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def cancel_btn_clicked(self):
        self.close()

    def on_data(self, data):
        self.data = data
        try:
            self.parent.fileListWidget.addItem(data['item'])
            for shape in data['shapes']:
                if not self.parent.uniqLabelList.findItems(shape.label, Qt.MatchExactly):
                    self.parent.uniqLabelList.addItem(shape.label)
                    self.parent.uniqLabelList.sortItems()
        except Exception as e:
            logger.error(e)

    def start_import(self, data):
        self.load = data['load']

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.show()

        # Execution
        executor = ImageImportExecutor(data)
        self.run_thread(executor, self.finish_import, custom_progress=self.progress_bar)

    def finish_import(self):
        self.parent.openNextImg(load=self.load)
        self.close()


class ImageImportExecutor(WorkerExecutor):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        logger.debug('Prepare training')

        try:
            import ptvsd
            ptvsd.debug_this_thread()
        except:
            pass

        data = Map(self.data)
        num_images = len(data.images)
        pattern = data.pattern
        output_dir = data.output_dir

        for i, filename in enumerate(data.images):

            self.thread.update.emit(_('Importing image {}/{}').format(i+1, num_images), i, num_images)
            
            if pattern and pattern not in filename:
                continue
            label_file = os.path.splitext(filename)[0] + '.json'
            if output_dir:
                label_file_without_path = os.path.basename(label_file)
                label_file = os.path.normpath(os.path.join(output_dir, label_file_without_path))
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

            shapes = []
            labelFile = LabelFile(label_file)
            for label, points, line_color, fill_color, shape_type, flags in labelFile.shapes:
                shape = Shape(label=label, shape_type=shape_type)
                shapes.append(shape)

            self.thread.data.emit({
                'item': item,
                'shapes': shapes,
            })
