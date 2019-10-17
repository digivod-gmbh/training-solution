import os
import re
import time
import math
import traceback

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Training
from labelme.extensions.networks import Network
from labelme.extensions.thread import WorkerExecutor
from labelme.utils import deltree, WorkerDialog, confirm
from labelme.config import MessageType
from labelme.config import get_config
from labelme.config.export import Export


class TrainingProgressWindow(WorkerDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Training'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        self.training_has_started = False

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        sep_label_text = _('of')

        epoch_label = QtWidgets.QLabel(_('Epoch'))
        self.epoch_value = QtWidgets.QLabel('-')
        epoch_sep = QtWidgets.QLabel(sep_label_text)
        self.epoch_max_value = QtWidgets.QLabel('-')

        batch_label = QtWidgets.QLabel(_('Batch'))
        self.batch_value = QtWidgets.QLabel('-')
        batch_sep = QtWidgets.QLabel(sep_label_text)
        self.batch_max_value = QtWidgets.QLabel('-')

        self.average_speed = 0
        self.speed_count = 0
        speed_label = QtWidgets.QLabel(_('Speed'))
        self.speed_value = QtWidgets.QLabel('-')

        time_label = QtWidgets.QLabel(_('Duration'))
        self.time_value = QtWidgets.QLabel('00:00:00')

        finished_label = QtWidgets.QLabel(_('Time left'))
        self.finished_value = QtWidgets.QLabel('-')

        self.start_time = False
        self.time_timer = QtCore.QTimer()
        self.time_timer.timeout.connect(self.timer_tick)
        self.time_timer.start(500)

        details_group = QtWidgets.QGroupBox()
        details_group.setTitle(_('Details'))
        details_group_layout = QtWidgets.QGridLayout()
        details_group.setLayout(details_group_layout)
        details_group_layout.addWidget(epoch_label, 0, 0, 1, 2)
        details_group_layout.addWidget(self.epoch_value, 0, 2)
        details_group_layout.addWidget(epoch_sep, 0, 3)
        details_group_layout.addWidget(self.epoch_max_value, 0, 4)
        details_group_layout.addWidget(batch_label, 1, 0, 1, 2)
        details_group_layout.addWidget(self.batch_value, 1, 2)
        details_group_layout.addWidget(batch_sep, 1, 3)
        details_group_layout.addWidget(self.batch_max_value, 1, 4)
        details_group_layout.addWidget(speed_label, 2, 0, 1, 2)
        details_group_layout.addWidget(self.speed_value, 2, 2, 1, 3)
        details_group_layout.addWidget(time_label, 3, 0, 1, 2)
        details_group_layout.addWidget(self.time_value, 3, 2, 1, 3)
        details_group_layout.addWidget(finished_label, 4, 0, 1, 2)
        details_group_layout.addWidget(self.finished_value, 4, 2, 1, 3)
        layout.addWidget(details_group)

        self.metric_labels = []
        self.metric_values = []
        self.metric_group = QtWidgets.QGroupBox()
        self.metric_group.setTitle(_('Metric'))
        self.metric_group_layout = QtWidgets.QGridLayout()
        self.metric_group.setLayout(self.metric_group_layout)
        layout.addWidget(self.metric_group)

        self.validation_labels = []
        self.validation_values = []
        self.validation_group = QtWidgets.QGroupBox()
        self.validation_group.setTitle(_('Last validation'))
        self.validation_group_layout = QtWidgets.QGridLayout()
        self.validation_group.setLayout(self.validation_group_layout)
        layout.addWidget(self.validation_group)

        self.image_label = QtWidgets.QLabel()
        layout.addWidget(self.image_label)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox()
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

        self.resize(300, 300)

    def timer_tick(self):
        if self.start_time is not False:
            current_time = time.time()
            seconds = current_time - self.start_time
            self.time_value.setText(self.format_duration(seconds))

    def format_duration(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

    def cancel_btn_clicked(self):
        self.close()
    
    def closeEvent(self, event):
        if self.current_worker_idx is not None:
            msg = _('Are you sure to cancel to current training session? The last completed epoch will be saved. Remember: Shorter training time leeds to worse results.')
            result = confirm(self, _('Training'), msg, MessageType.Warning)
            if result:
                super().closeEvent(event)
            else:
                event.ignore()
        else:
            event.accept()

    def on_data(self, data):
        self.data = data
        data = Map(data)
        try:
            if 'validation' in data:
                num_new_items = len(data.validation.items())
                num_old_items = len(self.validation_values)

                row = 0
                for item in data.validation.items():
                    if row >= num_old_items:
                        label = QtWidgets.QLabel(str(item[0]))
                        value = QtWidgets.QLabel('-')
                        value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                        self.validation_group_layout.addWidget(label, row, 0)
                        self.validation_group_layout.addWidget(value, row, 1)
                        self.validation_labels.append(label)
                        self.validation_values.append(value)
                    self.validation_labels[row].setText(str(item[0]))
                    if isinstance(item[1], (int, float)):
                        self.validation_values[row].setText('{:.4f}'.format(item[1]))
                    else:
                        self.validation_values[row].setText(str(item[1]))
                    row += 1

                if num_old_items > num_new_items:
                    for i in range(num_new_items, num_old_items):
                        self.validation_labels[i].setText('')
                        self.validation_values[i].setText('')
                        self.validation_group_layout.removeWidget(self.validation_labels[i])
                        self.validation_group_layout.removeWidget(self.validation_values[i])
                    for i in range(num_old_items, num_new_items, -1):
                        del self.validation_labels[i - 1]
                        del self.validation_values[i - 1]

            if 'progress' in data:
                progress = Map(data.progress)
                if 'epoch' in progress:
                    self.epoch_value.setText(str(progress.epoch))
                if 'epoch_max' in progress:
                    self.epoch_max_value.setText(str(progress.epoch_max))
                if 'batch' in progress:
                    self.batch_value.setText(str(progress.batch))
                if 'batch_max' in progress:
                    self.batch_max_value.setText(str(progress.batch_max))
                if 'speed' in progress:
                    self.speed_value.setText('{:.2f} {}'.format(progress.speed, _('samples/sec')))
                
                if not self.training_has_started:
                    self.start_time = time.time()
                    self.training_has_started = True

                if 'metric' in progress:
                    num_new_items = len(progress.metric.items())
                    num_old_items = len(self.metric_values)

                    row = 0
                    for item in progress.metric.items():
                        if row >= num_old_items:
                            label = QtWidgets.QLabel(str(item[0]))
                            value = QtWidgets.QLabel('-')
                            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                            self.metric_group_layout.addWidget(label, row, 0)
                            self.metric_group_layout.addWidget(value, row, 1)
                            self.metric_labels.append(label)
                            self.metric_values.append(value)
                        self.metric_labels[row].setText(str(item[0]))
                        if isinstance(item[1], (int, float)):
                            self.metric_values[row].setText('{:.4f}'.format(item[1]))
                        else:
                            self.metric_values[row].setText(str(item[1]))
                        row += 1

                    if num_old_items > num_new_items:
                        for i in range(num_new_items, num_old_items):
                            self.metric_labels[i].setText('')
                            self.metric_values[i].setText('')
                            self.metric_group_layout.removeWidget(self.metric_labels[i])
                            self.metric_group_layout.removeWidget(self.metric_values[i])
                        for i in range(num_old_items, num_new_items, -1):
                            del self.metric_labels[i - 1]
                            del self.metric_values[i - 1]

                # Estimate finish time
                if self.start_time is not False:
                    if 'epoch' in progress and 'epoch_max' in progress and 'batch' in progress and 'batch_max' in progress:
                        percentage = ((progress.epoch - 1) / progress.epoch_max) + (progress.batch / progress.batch_max / progress.epoch_max)
                        current_time = time.time()
                        duration = current_time - self.start_time
                        seconds_left = (duration / percentage) - duration
                        self.finished_value.setText(self.format_duration(seconds_left))

        except Exception as e:
            logger.error(traceback.format_exc())

    def start_training(self, data):
        config = get_config()

        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)

        if not data['val_dataset']:
            self.validation_group.hide()

        # Execution
        executor = TrainingExecutor(data)
        self.run_thread(executor, self.finish_training, custom_progress=self.progress_bar)

    def finish_training(self):
        data = self.data
        logger.debug('finish_training: {}'.format(data))
        self.progress_bar.setValue(4)

        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Training'), _('Network has been trained successfully'))
        self.reset_thread()
        self.close()

    def on_error(self, e):
        super().on_error(e)
        self.reset_thread()
        self.close()

    def on_abort(self):
        super().on_abort()
        self.reset_thread()
        self.close()


class TrainingExecutor(WorkerExecutor):

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

        network_key = self.data['network']
        if network_key not in Training.config('objects'):
            self.thread.message.emit(_('Training'), _('Network {} could not be found').format(network_key), MessageType.Error)
            self.abort()
            return

        # Training settings
        gpus = []
        gpu_checkboxes = self.data['gpu_checkboxes']
        for i, gpu in enumerate(gpu_checkboxes):
            if gpu.checkState() == Qt.Checked:
                gpus.append(str(i))
        gpus = ','.join(gpus)
        epochs = int(self.data['args_epochs'])
        batch_size = int(self.data['args_batch_size'])

        # Dataset
        dataset_format = self.data['selected_format']
        train_dataset_obj = Export.config('objects')[dataset_format]()
        train_dataset_obj.setInputFolderOrFile(self.data['train_dataset'])
        if self.data['val_dataset']:
            val_dataset_obj = Export.config('objects')[dataset_format]()
            val_dataset_obj.setInputFolderOrFile(self.data['val_dataset'])

        labels = train_dataset_obj.getLabels()
        num_train_samples = train_dataset_obj.getNumSamples()
        num_batches = int(math.ceil(num_train_samples / batch_size))
        
        args = Map({
            'network': self.data['network'],
            'train_dataset': self.data['train_dataset'],
            'validate_dataset': self.data['val_dataset'],
            'training_name': self.data['training_name'],
            'batch_size': batch_size,
            'learning_rate': float(self.data['args_learning_rate']),
            'gpus': gpus,
            'epochs': epochs,
            'early_stop_epochs': int(self.data['args_early_stop_epochs']),
            'start_epoch': self.data['start_epoch'],
            'resume': self.data['resume_training'],
        })

        self.thread.update.emit(_('Loading data ...'), 0, epochs * num_batches + 5)

        with Training.config('objects')[network_key]() as network:
            network.setAbortable(self.abortable)
            network.setThread(self.thread)
            network.setArgs(args)
            network.setOutputFolder(self.data['output_folder'])
            network.setTrainDataset(train_dataset_obj, dataset_format)
            network.setLabels(labels)

            if self.data['val_dataset']:
                network.setValDataset(val_dataset_obj)

            self.checkAborted()

            network.training()
