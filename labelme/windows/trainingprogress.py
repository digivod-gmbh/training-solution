import os
import re
import time
import math

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Training
from labelme.extensions.networks import Network
from labelme.extensions.thread import WorkerExecutor
from labelme.utils import deltree, WorkerDialog
from labelme.config import MessageType
from labelme.config import get_config
from labelme.config.export import Export


class TrainingProgressWindow(WorkerDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_('Training'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

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

        self.metric_values = []
        self.metric_group = QtWidgets.QGroupBox()
        self.metric_group.setTitle(_('Metric'))
        self.metric_group_layout = QtWidgets.QGridLayout()
        self.metric_group.setLayout(self.metric_group_layout)
        layout.addWidget(self.metric_group)

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
        mb = QtWidgets.QMessageBox()
        clicked_btn = mb.warning(self, _('Training'), _('Are you sure to cancel to current training session? The last completed epoch will be saved. Remember: Shorter training time leeds to worse results.'), QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if clicked_btn == QtWidgets.QMessageBox.Yes:
            super().closeEvent(event)
        else:
            event.ignore()

    def on_data(self, data):
        self.data = data
        data = Map(data)
        try:
            if 'validation' in data:
                if len(self.validation_values) == 0:
                    row = 0
                    for item in data.validation.items():
                        label = QtWidgets.QLabel(str(item[0]))
                        value = QtWidgets.QLabel('-')
                        value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                        self.validation_values.append(value)
                        self.validation_group_layout.addWidget(label, row, 0)
                        self.validation_group_layout.addWidget(value, row, 1)
                        row += 1
                row = 0
                for item in data.validation.items():
                    self.validation_values[row].setText('{:.4f}'.format(item[1]))
                    row += 1

            if 'progress' in data:
                progress = Map(data.progress)
                self.epoch_value.setText(str(progress.epoch))
                self.epoch_max_value.setText(str(progress.epoch_max))
                self.batch_value.setText(str(progress.batch))
                self.batch_max_value.setText(str(progress.batch_max))
                self.speed_value.setText('{:.2f} {}'.format(progress.speed, _('samples/sec')))
                if len(self.metric_values) == 0:
                    row = 0
                    for item in progress.metric.items():
                        label = QtWidgets.QLabel(str(item[0]))
                        value = QtWidgets.QLabel('-')
                        value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                        self.metric_values.append(value)
                        self.metric_group_layout.addWidget(label, row, 0)
                        self.metric_group_layout.addWidget(value, row, 1)
                        row += 1
                    self.start_time = time.time()
                row = 0
                for item in progress.metric.items():
                    self.metric_values[row].setText('{:.4f}'.format(item[1]))
                    row += 1

                # Estimate finish time
                if self.start_time is not False:
                    percentage = ((progress.epoch - 1) / progress.epoch_max) + (progress.batch / progress.batch_max / progress.epoch_max)
                    current_time = time.time()
                    duration = current_time - self.start_time
                    seconds_left = (duration / percentage) - duration
                    self.finished_value.setText(self.format_duration(seconds_left))

        except Exception as e:
            logger.error(e)

    def start_training(self, data):
        config = get_config()

        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)

        # Execution
        executor = TrainingExecutor(data)
        self.run_thread(executor, self.finish_training, custom_progress=self.progress_bar)

    def finish_training(self):
        data = self.data
        logger.debug(data)
        self.progress_bar.setValue(4)

        mb = QtWidgets.QMessageBox()
        mb.information(self, _('Training'), _('Network has been trained successfully'))
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

        create_dataset = self.data['create_dataset']
        if create_dataset:
            export_data = self.data['dataset_export_data']
            format_name = export_data['format']
            dataset_format = Export.config('objects')[format_name]()
            output_folder = export_data['output_folder']
            train_file = dataset_format.getOutputFileName('train')
            train_dataset = os.path.join(output_folder, train_file)
            validation_ratio = export_data['validation_ratio']
            if validation_ratio > 0:
                val_file = dataset_format.getOutputFileName('val')
                val_dataset = os.path.join(output_folder, val_file)
            else:
                # Validation dataset is optional
                val_dataset = False

        else:
            train_dataset = self.data['train_dataset']
            is_train_dataset_valid = True
            if not train_dataset:
                is_train_dataset_valid = False
            train_dataset = os.path.normpath(train_dataset)
            if not (os.path.isdir(train_dataset) or os.path.isfile(train_dataset)):
                is_train_dataset_valid = False
            if not is_train_dataset_valid:
                self.thread.message.emit(_('Training'), _('Please select a valid training dataset'), MessageType.Warning)
                self.abort()
                return

            val_dataset = self.data['val_dataset']
            is_val_dataset_valid = True
            if not val_dataset:
                is_val_dataset_valid = False
            val_dataset = os.path.normpath(val_dataset)
            if not (os.path.isdir(val_dataset) or os.path.isfile(val_dataset)):
                is_val_dataset_valid = False
            if not is_val_dataset_valid:
                # Validation dataset is optional
                val_dataset = False

        output_folder = os.path.normpath(self.data['output_folder'])
        training_name = self.data['training_name']
        training_name = re.sub(r'[^a-zA-Z0-9 _-]+', '', training_name)

        if not training_name:
            self.thread.message.emit(_('Training'), _('Please enter a valid training name'), MessageType.Warning)
            self.abort()
            return
        
        output_folder = os.path.join(output_folder, training_name)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        elif len(os.listdir(output_folder)) > 0:
            msg = _('The selected output directory "{}" is not empty. All containing files will be deleted. Are you sure to continue?').format(output_folder)
            if self.doConfirm(_('Training'), msg, MessageType.Warning):
                deltree(output_folder)
                time.sleep(0.5) # wait for deletion to be finished
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                self.abort()
                return

        if not os.path.isdir(output_folder):
            self.thread.message.emit(_('Training'), _('The selected output directory "{}" could not be created').format(output_folder), MessageType.Warning)
            self.abort()
            return

        network = self.data['network']

        networks = Training.config('networks')
        func_name = None
        for key in networks:
            if network in networks[key]:
                func_name = key
        
        if func_name is None:
            self.thread.message.emit(_('Training'), _('Network {} could not be found').format(network), MessageType.Error)
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
        train_dataset_obj.setInputFolderOrFile(train_dataset)
        if val_dataset:
            val_dataset_obj = Export.config('objects')[dataset_format]()
            val_dataset_obj.setInputFolderOrFile(val_dataset)

        labels = train_dataset_obj.getLabels()
        num_train_samples = train_dataset_obj.getNumSamples()
        num_batches = int(math.ceil(num_train_samples / batch_size))
        
        args = Map({
            'train_dataset': train_dataset,
            'validate_dataset': val_dataset,
            'training_name': training_name,
            'batch_size': batch_size,
            'learning_rate': float(self.data['args_learning_rate']),
            'gpus': gpus,
            'epochs': epochs,
        })

        self.thread.update.emit(_('Loading data ...'), 0, epochs * num_batches + 5)

        network = Training.config('objects')[func_name]()
        network.setAbortable(self.abortable)
        network.setThread(self.thread)
        network.setArgs(args)
        network.setOutputFolder(output_folder)
        network.setTrainDataset(train_dataset_obj)
        network.setLabels(labels)

        if val_dataset:
            network.setValDataset(val_dataset_obj)

        self.checkAborted()

        network.training()

