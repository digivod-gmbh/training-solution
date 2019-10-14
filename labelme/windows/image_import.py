import time
import os
import re

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.extensions.thread import WorkerExecutor
from labelme.utils import WorkerDialog, Application, StatisticsModel
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

        self.status_label = QtWidgets.QLabel(_('Loading ...'))
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        button_box = QtWidgets.QDialogButtonBox()
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

    def cancel_btn_clicked(self):
        self.status_label.setText(_('Cancelling ...'))
        self.on_abort()
        self.parent.statistics_widget.clear()
        self.parent.statistics_widget.setRowCount(0)
        self.parent.fileListWidget.clear()
        self.parent.uniqLabelList.clear()
        self.parent.resetState()
        self.close()

    def start_import(self, data):
        self.load = data['load']
        self.initial = data['initial']

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.show()

        self.parent.status(_('Loading images ...'))

        data['update_interval'] = int(self.parent._config['statistics_update_interval'])

        # Execution
        executor = ImageImportExecutor(data)
        self.run_thread(executor, self.finish_import, custom_progress=self.progress_bar)

    def finish_import(self):
        if self.initial:
            self.parent.updateLabelHistory()
            labels = self.parent.statistics_model.getLabels()
            self.parent.labelFilter.clear()
            self.parent.labelFilter.addItem(_('- all labels -'), StatisticsModel.STATISTICS_FILTER_ALL)
            for label in labels:
                self.parent.labelFilter.addItem(label, label)
        self.parent.openNextImg(load=self.load)
        self.close()

    def on_data(self, data):
        self.data = data
        model = self.parent.statistics_model
        model.addImages(data['num_images'], data['all_shapes'])

        for item in data['items']:
            self.parent.fileListWidget.addItem(item)

        for shapes in data['all_shapes']:
            for shape in shapes:
                if not self.parent.uniqLabelList.findItems(shape.label, Qt.MatchExactly):
                    self.parent.uniqLabelList.addItem(shape.label)
        self.parent.uniqLabelList.sortItems()


class ImageImportExecutor(WorkerExecutor):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        logger.debug('Start import from directory')

        try:
            import ptvsd
            ptvsd.debug_this_thread()
        except:
            pass

        data = Map(self.data)
        num_images = len(data.images)
        pattern = data.pattern
        output_dir = data.output_dir
        filters = data.filters

        filter_label_func = self.acceptAll
        if 'label' in filters and not filters['label'] == StatisticsModel.STATISTICS_FILTER_ALL:
            filter_label_func = self.acceptLabel

        image_count = 0
        all_shapes = []
        items = []

        self.checkAborted()

        for i, filename in enumerate(data.images):

            self.thread.update.emit(None, i, num_images)
            self.checkAborted()

            # Search pattern
            if pattern and pattern.lower() not in filename.lower(): # re.search(pattern, filename, re.IGNORECASE) == None:
                continue

            label_file = os.path.splitext(filename)[0] + '.json'
            if output_dir:
                label_file_without_path = os.path.basename(label_file)
                label_file = os.path.normpath(os.path.join(output_dir, label_file_without_path))

            # ListItem
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Unchecked)

            self.checkAborted()

            shapes = []
            has_labels = False
            labels_for_image = set([])
            label_file_exists = os.path.isfile(label_file)

            # Labels
            if label_file_exists:
                labelFile = LabelFile(label_file)
                for label, points, line_color, fill_color, shape_type, flags in labelFile.shapes:
                    if filter_label_func(label):
                        has_labels = True
                        shape = Shape(label=label, shape_type=shape_type)
                        shapes.append(shape)
                        labels_for_image.add(label)

            # Filters
            if 'label' in filters and not filters['label'] == StatisticsModel.STATISTICS_FILTER_ALL:
                if not filters['label'] in labels_for_image:
                    continue
            if 'has_label' in filters:
                if filters['has_label'] == StatisticsModel.STATISTICS_FILTER_LABELED and not has_labels:
                    continue
                if filters['has_label'] == StatisticsModel.STATISTICS_FILTER_UNLABELED and has_labels:
                    continue

            image_count += 1
            items.append(item)
            if has_labels:
                item.setCheckState(Qt.Checked)
                all_shapes.append(shapes)

            if image_count % data['update_interval'] == 0:
                self.thread.data.emit({
                    'items': items,
                    'num_images': image_count,
                    'all_shapes': all_shapes,
                })
                image_count = 0
                all_shapes = []
                items = []

            self.checkAborted()

        self.thread.data.emit({
            'num_images': image_count,
            'all_shapes': all_shapes,
            'items': items,
        })

    def acceptAll(self, *args):
        return True

    def acceptLabel(self, label):
        return self.data['filters']['label'] == label

