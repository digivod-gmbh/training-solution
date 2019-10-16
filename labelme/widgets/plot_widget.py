import pyqtgraph as pg

class PlotWidget(pg.PlotWidget):

    def __init__(self, parent=None, **kwargs):

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        super().__init__(parent)

        if 'left_label' in kwargs:
            self.setLabel('left', kwargs['left_label'], angle=90)
        if 'bottom_label' in kwargs:
            self.setLabel('bottom', kwargs['bottom_label'])
        if 'x_range' in kwargs:
            self.setXRange(0, kwargs['x_range'])
        if 'y_range' in kwargs:
            self.setYRange(0, kwargs['y_range'])

        self.data_y = [0.9, 0.89, 0.85, 0.6, 0.2, 0.45, 0.5, 0.5, 0.55, 0.8, 0.9, 0.1]
        self.data_x = list(range(1, len(self.data_y) + 1))

        self.plot(self.data_x, self.data_y, pen='r')

    def update(self, value, label=None):
        self.data_y.append(value)
        if label is not None:
            self.data_x.append(label)
        else:
            self.data_x.append(self.data_x[-1] + 1)
        self.plot(self.data_x, self.data_y, pen='b', clear=True)
