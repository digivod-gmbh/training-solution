import os

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Export, Training
from labelme.extensions.networks import Network
from labelme.utils import Worker, ProgressObject, Application
from labelme.config import get_config


class InferenceWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent
        
        super().__init__(parent)
        self.setWindowTitle(_('Validation'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.image_label = QtWidgets.QLabel()
        layout.addWidget(self.image_label)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

    def get_scaled_size(self, w, h, max_size):
        if w > max_size or h > max_size:
            if w > h:
                ratio = max_size / w
                w = max_size
                h = h * ratio
            else:
                ratio = max_size / h
                h = max_size
                w = w * ratio
        return w, h

    def start_inference(self, input_image_file, training_folder):
        config = get_config()
        self.pixmap = QtGui.QPixmap(input_image_file)
        w, h = self.get_scaled_size(self.pixmap.width(), self.pixmap.height(), config['image_max_size'])
        self.pixmap = self.pixmap.scaled(w, h)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.show()

        self.painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(QtCore.Qt.red)
        pen.setWidth(2)
        self.painter.setPen(pen)

        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)

        config_file = os.path.join(training_folder, Training.config('config_file'))

        self.network = Network()
        network_config = self.network.loadConfig(config_file)

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

        self.inference_data = Map({
            'input_image_file': input_image_file,
            'label_file': label_file,
            'architecture_file': architecture_file,
            'weights_file': weights_file,
            'labels': network_config.labels,
        })

        worker_idx, worker = Application.createWorker()
        self.worker_idx = worker_idx
        self.worker_object = ProgressObject(worker, self.inference, self.error_progress, self.network.abort, 
            self.update_progress, data_func=self.receive_data)
        self.network.setThread(self.worker_object)

        self.progress_bar.setValue(1)

        worker.addObject(self.worker_object)
        worker.start()

    def inference(self):
        self.network.inference(self.inference_data.input_image_file, self.inference_data.label_file, 
            self.inference_data.architecture_file, self.inference_data.weights_file, args = None)

    def receive_data(self, data):
        self.progress_bar.setValue(4)
        data = Map(data)
        for i in range(len(data.bbox[0])):
            label = int(data.classid[0][i][0])
            score = data.score[0][i][0]
            if label > -1:
                label_name = _('unknown')
                if label < len(self.inference_data.labels):
                    label_name = str(self.inference_data.labels[label])
                xr = self.pixmap.width() / data.imgsize[0]
                yr = self.pixmap.height() / data.imgsize[1]
                x, y = data.bbox[0][i][0] * xr, data.bbox[0][i][1] * yr
                w, h = data.bbox[0][i][2] * xr - x, data.bbox[0][i][3] * yr - y
                logger.debug('Draw bbox ({}, {}, {}, {}) for label {} ({})'.format(int(x), int(y), int(w), int(h), label_name, label))
                self.painter.scale(1, 1)
                self.painter.drawRect(x, y, w, h)
                p1 = QtCore.QPointF(x + 4, y + 12)
                p2 = QtCore.QPointF(x + 4, y + 24)
                self.painter.drawText(p1, label_name)
                self.painter.drawText(p2, '{0:.4f}'.format(score))
        self.image_label.setPixmap(self.pixmap)
        self.image_label.show()

    def update_progress(self, msg=None, value=None):
        if value is not None:
            self.progress_bar.setValue(value)
        if value == -1:
            val = self.progress_bar.value() + 1
            self.progress_bar.setValue(val)

    def error_progress(self, e):
        mb = QtWidgets.QMessageBox()
        mb.warning(self, _('Validation'), _('An error occured during validation of training'))