import os

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Training
from labelme.extensions.networks import Network
from labelme.extensions.thread import WorkerExecutor
from labelme.utils import WorkerDialog
from labelme.config import MessageType
from labelme.config import get_config
from labelme.config.export import Export


class InferenceWindow(WorkerDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Validation'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.NonModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.image_label = QtWidgets.QLabel()
        layout.addWidget(self.image_label)

        self.score_value = QtWidgets.QLabel('50%')
        self.score_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.score_slider.setMinimum(0)
        self.score_slider.setMaximum(100)
        self.score_slider.setSingleStep(1)
        self.score_slider.setValue(50)
        self.score_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.score_slider.setTickInterval(10)
        self.score_slider.valueChanged.connect(self.update_results)

        self.confidence_group = QtWidgets.QGroupBox()
        self.confidence_group.setTitle(_('Minimum confidence'))
        confidence_group_layout = QtWidgets.QGridLayout()
        self.confidence_group.setLayout(confidence_group_layout)
        confidence_group_layout.addWidget(self.score_value, 0, 0)
        confidence_group_layout.addWidget(self.score_slider, 0, 1, 1, 8)
        layout.addWidget(self.confidence_group)
        self.confidence_group.hide()

        layout.addStretch()

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

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

        # Data
        data = {
            'training_folder': training_folder,
            'input_image_file': input_image_file,
        }
        self.input_image_file = input_image_file

        # Execution
        executor = InferenceExecutor(data)
        self.run_thread(executor, self.finish_inference, custom_progress=self.progress_bar)

    def finish_inference(self):
        data = self.data
        logger.debug(data)
        self.confidence_group.show()
        self.progress_bar.setValue(4)
        self.update_results()

    def reset_image(self):
        config = get_config()
        pixmap = QtGui.QPixmap(self.input_image_file)
        w, h = self.get_scaled_size(pixmap.width(), pixmap.height(), config['image_max_size'])
        scaled_pixmap = pixmap.scaled(w, h)
        self.painter.drawPixmap(0, 0, scaled_pixmap)

    def update_results(self):
        data = Map(self.data)
        min_score = self.score_slider.value() / 100.0
        self.score_value.setText('{}%'.format(self.score_slider.value()))
        self.reset_image()
        for i in range(len(data.bbox[0])):
            label = int(data.classid[0][i][0])
            score = data.score[0][i][0]
            if label > -1 and score > min_score:
                label_name = _('unknown')
                if label < len(data.labels):
                    label_name = str(data.labels[label])
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


class InferenceExecutor(WorkerExecutor):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        logger.debug('Prepare inference')

        try:
            import ptvsd
            ptvsd.debug_this_thread()
        except:
            pass

        training_folder = self.data['training_folder']
        input_image_file = self.data['input_image_file']

        config_file = os.path.join(training_folder, Training.config('config_file'))

        network = Network()
        network.setAbortable(self.abortable)
        network.setThread(self.thread)

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

        inference_data = Map({
            'input_image_file': input_image_file,
            'architecture_file': architecture_file,
            'weights_file': weights_file,
            'labels': network_config.labels,
        })

        network.inference(inference_data.input_image_file, inference_data.labels, 
            inference_data.architecture_file, inference_data.weights_file, args = None)
        