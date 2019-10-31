import os

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Training
from labelme.extensions.networks import Network
from labelme.utils import WorkerExecutor
from labelme.utils import WorkerDialog
from labelme.config import MessageType
from labelme.config import get_config
from labelme.config.export import Export
from labelme.widgets import HelpLabel, HelpCheckbox, HelpGroupBox


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

        self.confidence_group = HelpGroupBox('Inference_Confidence', _('Minimum confidence'))
        confidence_group_layout = QtWidgets.QGridLayout()
        self.confidence_group.widget.setLayout(confidence_group_layout)
        confidence_group_layout.addWidget(self.score_value, 0, 0)
        confidence_group_layout.addWidget(self.score_slider, 0, 1, 1, 8)
        layout.addWidget(self.confidence_group)
        self.confidence_group.hide()

        # Network config
        config_training_name_label = QtWidgets.QLabel(_('Training name'))
        self.config_training_name_value = QtWidgets.QLabel('-')

        config_network_label = QtWidgets.QLabel(_('Network'))
        self.config_network_value = QtWidgets.QLabel('-')

        config_epochs_label = QtWidgets.QLabel(_('Epochs'))
        self.config_epochs_value = QtWidgets.QLabel('-')

        config_trained_epochs_label = HelpLabel('Inference_Config_Epochs', _('Trained epochs'))
        self.config_trained_epochs_value = QtWidgets.QLabel('-')

        config_early_stop_label = QtWidgets.QLabel(_('Early stop epochs'))
        self.config_early_stop_value = QtWidgets.QLabel('-')

        config_batch_size_label = QtWidgets.QLabel(_('Batch size'))
        self.config_batch_size_value = QtWidgets.QLabel('-')

        config_learning_rate_label = QtWidgets.QLabel(_('Learning rate'))
        self.config_learning_rate_value = QtWidgets.QLabel('-')

        config_dataset_format_label = QtWidgets.QLabel(_('Dataset format'))
        self.config_dataset_format_value = QtWidgets.QLabel('-')

        config_dataset_train_label = QtWidgets.QLabel(_('Train dataset'))
        self.config_dataset_train_value = QtWidgets.QLabel('-')

        config_dataset_val_label = QtWidgets.QLabel(_('Validation dataset'))
        self.config_dataset_val_value = QtWidgets.QLabel('-')

        config_group = HelpGroupBox('Inference_NetworkConfig', _('Network config'))
        config_group_layout = QtWidgets.QGridLayout()
        config_group.widget.setLayout(config_group_layout)
        config_group_layout.addWidget(config_training_name_label, 0, 0)
        config_group_layout.addWidget(self.config_training_name_value, 0, 1)
        config_group_layout.addWidget(config_network_label, 1, 0)
        config_group_layout.addWidget(self.config_network_value, 1, 1)
        config_group_layout.addWidget(config_epochs_label, 2, 0)
        config_group_layout.addWidget(self.config_epochs_value, 2, 1)
        config_group_layout.addWidget(config_trained_epochs_label, 3, 0)
        config_group_layout.addWidget(self.config_trained_epochs_value, 3, 1)
        config_group_layout.addWidget(config_early_stop_label, 4, 0)
        config_group_layout.addWidget(self.config_early_stop_value, 4, 1)
        config_group_layout.addWidget(config_batch_size_label, 5, 0)
        config_group_layout.addWidget(self.config_batch_size_value, 5, 1)
        config_group_layout.addWidget(config_learning_rate_label, 6, 0)
        config_group_layout.addWidget(self.config_learning_rate_value, 6, 1)
        config_group_layout.addWidget(config_dataset_format_label, 7, 0)
        config_group_layout.addWidget(self.config_dataset_format_value, 7, 1)
        config_group_layout.addWidget(config_dataset_train_label, 8, 0)
        config_group_layout.addWidget(self.config_dataset_train_value, 8, 1)
        config_group_layout.addWidget(config_dataset_val_label, 9, 0)
        config_group_layout.addWidget(self.config_dataset_val_value, 9, 1)
        
        layout.addWidget(config_group)

        layout.addStretch()

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

        # Data
        data = {
            'training_folder': training_folder,
            'input_image_file': input_image_file,
        }
        self.input_image_file = input_image_file

        # Execution
        executor = InferenceExecutor(data)
        self.run_thread(executor, self.finish_inference)

    def finish_inference(self):
        data = self.data
        logger.debug(data)
        self.confidence_group.show()
        self.update_results()

    def reset_image(self):
        config = get_config()
        pixmap = QtGui.QPixmap(self.input_image_file)
        w, h = self.get_scaled_size(pixmap.width(), pixmap.height(), config['image_max_size'])
        scaled_pixmap = pixmap.scaled(w, h)
        self.painter.drawPixmap(0, 0, scaled_pixmap)

    def on_data(self, data):
        super().on_data(data)
        if 'network_config' in data:
            self.update_config_info(data['network_config'])

    def update_config_info(self, config):
        if 'network' in config:
            network_name = config['network']
            if network_name in Training.config('networks'):
                network_name = Training.config('networks')[network_name]
            self.config_network_value.setText(network_name)
        if 'training_name' in config['args']:
            self.config_training_name_value.setText(str(config['args']['training_name']))
        if 'epochs' in config['args']:
            self.config_epochs_value.setText(str(config['args']['epochs']))
        if 'last_epoch' in config:
            self.config_trained_epochs_value.setText(str(config['last_epoch']))
        if 'early_stop_epochs' in config['args']:
            self.config_early_stop_value.setText(str(config['args']['early_stop_epochs']))
        if 'batch_size' in config['args']:
            self.config_batch_size_value.setText(str(config['args']['batch_size']))
        if 'learning_rate' in config['args']:
            self.config_learning_rate_value.setText(str(config['args']['learning_rate']))
        if 'dataset' in config:
            dataset_format = config['dataset']
            if dataset_format in Export.config('formats'):
                dataset_format = Export.config('formats')[dataset_format]
            self.config_dataset_format_value.setText(dataset_format)

        project_folder = self.parent.parent.settings.value('settings/project/folder', '')
        logger.debug('Restored value "{}" for setting settings/project/folder'.format(project_folder))

        if 'train_dataset' in config['args']:
            train_dataset = config['args']['train_dataset']
            prefix = os.path.commonprefix([train_dataset, project_folder])
            if len(prefix) >= len(project_folder):
                train_dataset = _('Project folder') + train_dataset[len(prefix):]
            self.config_dataset_train_value.setText(train_dataset)
        if 'validate_dataset' in config['args']:
            validate_dataset = config['args']['validate_dataset']
            prefix = os.path.commonprefix([validate_dataset, project_folder])
            if len(prefix) >= len(project_folder):
                validate_dataset = _('Project folder') + validate_dataset[len(prefix):]
            self.config_dataset_val_value.setText(validate_dataset)

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
                xr = self.pixmap.width() / data.imgsize[1]
                yr = self.pixmap.height() / data.imgsize[0]
                x, y = data.bbox[0][i][0] * xr, data.bbox[0][i][1] * yr
                w, h = data.bbox[0][i][2] * xr - x, data.bbox[0][i][3] * yr - y
                #logger.debug('Draw bbox ({}, {}, {}, {}) for label {} ({})'.format(int(x), int(y), int(w), int(h), label_name, label))
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
        self.thread.data.emit({'network_config': network_config})

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

        self.thread.update.emit(_('Validating ...'), 1, 3)

        network.inference(inference_data.input_image_file, inference_data.labels, 
            inference_data.architecture_file, inference_data.weights_file, args = None)

        self.thread.update.emit(_('Finished'), 3, 3)
        