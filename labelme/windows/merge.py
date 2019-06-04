import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from labelme.utils.map import Map
from labelme.logger import logger
from labelme.config import Export


class MergeWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        self.parent = parent

        super().__init__(parent)
        self.setWindowTitle(_('Merge datasets'))
        self.set_default_window_flags(self)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.dataset_file_table = QtWidgets.QTableWidget(0, 4)
        self.dataset_file_table.setHorizontalHeaderLabels([_('Dataset'), _('Format'), _('# Samples'), _('Directory')])
        header = self.dataset_file_table.horizontalHeader()
        header.setMinimumSectionSize(100)
        self.dataset_file_table.setColumnWidth(2, 160)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)

        dataset_files_browse_btn = QtWidgets.QPushButton(_('Add dataset file'))
        dataset_files_browse_btn.clicked.connect(self.dataset_files_browse_btn_clicked)

        dataset_files_group = QtWidgets.QGroupBox()
        dataset_files_group.setTitle(_('Datasets'))
        dataset_files_group_layout = QtWidgets.QGridLayout()
        dataset_files_group.setLayout(dataset_files_group_layout)
        dataset_files_group_layout.addWidget(self.dataset_file_table, 0, 0, 1, 4)
        dataset_files_group_layout.addWidget(dataset_files_browse_btn, 1, 0)
        layout.addWidget(dataset_files_group)

        self.export_file = QtWidgets.QLineEdit()
        export_browse_btn = QtWidgets.QPushButton(_('Browse'))
        export_browse_btn.clicked.connect(self.export_browse_btn_clicked)

        export_file_group = QtWidgets.QGroupBox()
        export_file_group.setTitle(_('Export folder'))
        export_file_group_layout = QtWidgets.QHBoxLayout()
        export_file_group.setLayout(export_file_group_layout)
        export_file_group_layout.addWidget(self.export_file)
        export_file_group_layout.addWidget(export_browse_btn)
        layout.addWidget(export_file_group)

        button_box = QtWidgets.QDialogButtonBox()
        merge_btn = button_box.addButton(_('Merge'), QtWidgets.QDialogButtonBox.AcceptRole)
        merge_btn.clicked.connect(self.merge_btn_clicked)
        cancel_btn = button_box.addButton(_('Cancel'), QtWidgets.QDialogButtonBox.RejectRole)
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        layout.addWidget(button_box)

        self.resize(750, 400)

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

    def dataset_files_browse_btn_clicked(self):
        # TODO: Replace config_file_extension
        filters = _('Dataset files') + ' (*{})'.format(Export.config('config_file_extension'))
        dataset_files, selected_filter = QtWidgets.QFileDialog.getOpenFileNames(self, _('Select dataset files'), '', filters)
        if len(dataset_files) > 0:
            for dataset_file in dataset_files:
                dataset_file = os.path.normpath(dataset_file)
                self.add_dataset_file(dataset_file)

    def export_browse_btn_clicked(self):
        last_dir = self.parent.settings.value('merge/last_export_dir', '')
        logger.debug('Restored value "{}" for setting merge/last_export_dir'.format(last_dir))
        # TODO: Replace config_file_extension
        filters = _('Dataset file') + ' (*{})'.format(Export.config('config_file_extension'))
        export_file, selected_filter = QtWidgets.QFileDialog.getSaveFileName(self, _('Save output file as'), last_dir, filters)
        if export_file:
            export_file = os.path.normpath(export_file)
            self.parent.settings.setValue('merge/last_export_dir', os.path.dirname(export_file))
            self.export_file.setText(export_file)

    def add_dataset_file(self, dataset_file):
        dataset_name = os.path.basename(dataset_file)
        dataset_path = os.path.normpath(dataset_file)
        num_samples = []
        dataset_data = Export.read_dataset_config(dataset_file)
        for key in dataset_data.samples:
            num_samples.append('{}={}'.format(key, dataset_data.samples[key]))
        num_samples = ', '.join(num_samples)
        dataset_format = Export.config('formats')[dataset_data.format]

        pos = self.dataset_file_table.rowCount()
        self.dataset_file_table.setRowCount(pos + 1)

        item_name = QtWidgets.QTableWidgetItem(dataset_name)
        item_name.setFlags(Qt.ItemIsEnabled)
        self.dataset_file_table.setItem(pos, 0, item_name)

        item_format = QtWidgets.QTableWidgetItem(dataset_format)
        item_format.setFlags(Qt.ItemIsEnabled)
        self.dataset_file_table.setItem(pos, 1, item_format)

        item_samples = QtWidgets.QTableWidgetItem(num_samples)
        item_samples.setFlags(Qt.ItemIsEnabled)
        self.dataset_file_table.setItem(pos, 2, item_samples)

        item_path = QtWidgets.QTableWidgetItem(dataset_path)
        item_path.setFlags(Qt.ItemIsEnabled)
        self.dataset_file_table.setItem(pos, 3, item_path)

    def merge_btn_clicked(self):
        mb = QtWidgets.QMessageBox
        export_file = self.export_file.text()
        export_dir = os.path.dirname(export_file)
        if not export_dir or not os.path.isdir(export_dir):
            mb.warning(self, _('Merge datasets'), _('Please enter a valid export folder'))
            return

        first_format = None
        dataset_files = []
        for row in range(self.dataset_file_table.rowCount()):
            current_format = self.dataset_file_table.item(row, 1).text()
            if first_format is None:
                first_format = current_format
            elif first_format != current_format:
                mb.warning(self, _('Merge datasets'), _('All dataset files must have the same format'))
                return
            dataset_files.append(self.dataset_file_table.item(row, 3).text())

        if len(dataset_files) < 2:
            mb.warning(self, _('Merge datasets'), _('Please select at least 2 datasets to merge'))
            return

        merge_datasets(dataset_files)

    def cancel_btn_clicked(self):
        self.close()

    def merge_datasets(datasets):
        
        #dataset_format = Export.config('objects')[func_name]()
        pass

