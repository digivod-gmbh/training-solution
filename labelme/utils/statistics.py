from qtpy import QtWidgets


class StatisticsModel():

    STATISTICS_SUB_ITEM_PREFIX = ' > '
    STATISTICS_FILTER_ALL = 'all'
    STATISTICS_FILTER_LABELED = 'labeled'
    STATISTICS_FILTER_UNLABELED = 'unlabeled'

    def __init__(self, widget):
        if not isinstance(widget, QtWidgets.QTableWidget):
            raise Exception('Statistics widget must be of type QTableWidget')
        self.widget = widget
        self.reset()

    def reset(self):
        self.num_images = 0
        self.num_labels = 0
        self.num_shapes = 0
        self.image_types = {
            'labeled': 0,
            'unlabeled': 0,
        }
        self.label_count = {}
        self.initWidget()

    def addImages(self, num_images, all_shapes):
        num_labeled = len(all_shapes)
        num_unlabeled = num_images - num_labeled
        self.num_images += num_images
        self.image_types['labeled'] += num_labeled
        self.image_types['unlabeled'] += num_unlabeled
        for shapes in all_shapes:
            self.addShapes(shapes)
        self.update()

    def addShapes(self, shapes):
        label_names = list(self.label_count.keys())
        for shape in shapes:
            label = shape.label
            if not label in label_names:
                self.label_count[label] = 0
                self.num_labels += 1
                label_names.append(label)
            self.label_count[label] += 1
            self.num_shapes += 1

    def addLabel(self, label, is_first_label):
        label_names = list(self.label_count.keys())
        if not label in label_names:
            self.label_count[label] = 0
            self.num_labels += 1
        self.label_count[label] += 1
        self.num_shapes += 1
        if is_first_label:
            self.image_types['labeled'] += 1
            self.image_types['unlabeled'] -= 1
        self.update()

    def remLabel(self, label, is_last_label):
        self.label_count[label] -= 1
        if self.label_count[label] == 0:
            del self.label_count[label]
            self.num_labels -= 1
        self.num_shapes -= 1
        if is_last_label:
            self.image_types['labeled'] -= 1
            self.image_types['unlabeled'] += 1
        self.update()
    
    def getLabels(self):
        return list(self.label_count.keys())

    def update(self):
        default_rows = 5
        row_count = default_rows + self.num_labels
        self.widget.clear()
        self.widget.setRowCount(row_count)
        self.widget.setColumnCount(2)

        self.widget.setItem(0, 0, QtWidgets.QTableWidgetItem(_('Images')))
        self.widget.setItem(0, 1, QtWidgets.QTableWidgetItem('{:n}'.format(self.num_images)))
        self.widget.setItem(1, 0, QtWidgets.QTableWidgetItem(StatisticsModel.STATISTICS_SUB_ITEM_PREFIX + _('Labeled')))
        self.widget.setItem(1, 1, QtWidgets.QTableWidgetItem('{:n}'.format(self.image_types['labeled'])))
        self.widget.setItem(2, 0, QtWidgets.QTableWidgetItem(StatisticsModel.STATISTICS_SUB_ITEM_PREFIX + _('Unlabeled')))
        self.widget.setItem(2, 1, QtWidgets.QTableWidgetItem('{:n}'.format(self.image_types['unlabeled'])))
        self.widget.setItem(3, 0, QtWidgets.QTableWidgetItem(_('Labels')))
        self.widget.setItem(3, 1, QtWidgets.QTableWidgetItem('{:n}'.format(self.num_labels)))
        self.widget.setItem(4, 0, QtWidgets.QTableWidgetItem(_('Annotations')))
        self.widget.setItem(4, 1, QtWidgets.QTableWidgetItem('{:n}'.format(self.num_shapes)))

        for i, label_name in enumerate(self.label_count.keys()):
            row = default_rows + i
            label_count = self.label_count[label_name]
            self.widget.setItem(row, 0, QtWidgets.QTableWidgetItem(StatisticsModel.STATISTICS_SUB_ITEM_PREFIX + str(label_name)))
            self.widget.setItem(row, 1, QtWidgets.QTableWidgetItem('{:n}'.format(label_count)))

        self.widget.resizeRowsToContents()

    def initWidget(self):
        self.widget.setColumnCount(2)
        self.widget.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        h_header = self.widget.horizontalHeader()
        h_header.setVisible(False)
        h_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        h_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        v_header = self.widget.verticalHeader()
        v_header.setVisible(False)
        self.update()
