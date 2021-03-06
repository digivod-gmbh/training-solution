import os
import random
import base64

from labelme.logger import logger
from labelme.label_file import LabelFile
from labelme.utils import WorkerExecutor


class IntermediateSample():

    def __init__(self, image, image_size, label, points, shape_type):
        self.image = image
        self.image_size = image_size
        self.label = label
        self.shape_type = shape_type # rectangle, polygon
        # Points format: [[p1_x, p1_y], [p2_x, p2_y], ...]
        # for bounding box: [[xmin, ymin], [xmax, ymax]]
        self.points = points

class IntermediateFormat(WorkerExecutor):
    
    def __init__(self):
        super().__init__()
        self.root = None
        self.images = set([])
        self.labels = set([])
        self.samples = []
        self.samplesPerLabel = {}
        self.included_labels = set([])
        self.validation_ratio = 0.0

    def addSample(self, image, image_size, label, points, shape_type):
        if len(self.included_labels) > 0 and label not in self.included_labels or len(points) == 0:
            return
        sample = IntermediateSample(image, image_size, label, points, shape_type)
        self.samples.append(sample)
        self.labels.add(label)
        self.images.add(image)
        if not label in self.samplesPerLabel:
            self.samplesPerLabel[label] = []
        self.samplesPerLabel[label].append(sample)

    def getTrainValidateSamples(self, shuffle=False, group_by_images=False):
        train_samples = []
        val_samples = []
        if group_by_images:
            train_samples = []
            val_samples = []
            num_train_samples = int(len(self.samples) * (1.0 - self.validation_ratio))
            samples_per_image = self.getSamplesPerImage()
            i = 0
            for image in samples_per_image:
                samples = samples_per_image[image]
                for sample in samples:
                    if i < num_train_samples:
                        train_samples.append(sample)
                    else:
                        val_samples.append(sample)
                i += len(samples)
                self.checkAborted()
        else:
            for label in self.labels:
                train_for_label, val_for_label = self.getTrainValidateSamplesPerLabel(label, shuffle)
                train_samples += train_for_label
                val_samples += val_for_label
                self.checkAborted()
        if shuffle:
            random.shuffle(train_samples)
            random.shuffle(val_samples)
        self.checkAborted()
        return train_samples, val_samples

    def getTrainValidateSamplesPerLabel(self, label, shuffle=False):
        if label not in self.samplesPerLabel:
            raise Exception('Label {} not found'.format(label))
        train_samples = self.samplesPerLabel[label]
        val_samples = []
        if shuffle:
            random.shuffle(train_samples)
        if self.validation_ratio > 0.0:
            num_val_samples = int(len(train_samples) * self.validation_ratio)
            logger.debug('Use {} validate samples for label {}'.format(num_val_samples, label))
            for i in range(num_val_samples):
                val_samples.append(train_samples.pop(-1))
                self.checkAborted()
        return train_samples, val_samples

    def getSamplesPerImage(self, samples=[]):
        if not samples:
            samples = self.samples
        samples_per_image = {}
        for sample in samples:
            if sample.image not in samples_per_image:
                samples_per_image[sample.image] = []
            samples_per_image[sample.image].append(sample)
            self.checkAborted()
        return samples_per_image

    def getLabelFilesFromDataFolder(self, data_folder):
        label_files = []
        for root, dirs, files in os.walk(data_folder):
            for f in files:
                if LabelFile.is_label_file(f):
                    full_path = os.path.normpath(os.path.join(root, f))
                    label_files.append(full_path)
                self.checkAborted()
        return label_files

    def addFromLabelFiles(self, data_folder, shuffle=False):
        self.root = data_folder
        label_files = self.getLabelFilesFromDataFolder(data_folder)
        for label_file in label_files:
            self.checkAborted()
            self.addFromLabelFile(label_file, root_folder=data_folder)
        if shuffle:
            for label_samples in self.samplesPerLabel:
                self.checkAborted()
                random.shuffle(self.samplesPerLabel[label_samples])

    def addFromLabelFile(self, label_file, root_folder):
        lf = LabelFile(label_file)
        image_size = (lf.imageHeight, lf.imageWidth)
        image_dir = os.path.dirname(os.path.relpath(label_file, root_folder))
        for s in lf.shapes:
            image_path = os.path.normpath(os.path.join(image_dir, lf.imagePath))
            self.addSample(image_path, image_size, s[0], s[1], s[4])
            self.checkAborted()

    def toLabelFiles(self):
        samples_per_image = self.getSamplesPerImage()
        num_images = len(samples_per_image)
        num_counter = 0
        for image in samples_per_image:
            samples = samples_per_image[image]
            self.toLabelFile(image, samples)
            percent = num_counter / num_images * 10 + 90
            self.thread.update.emit(_('Writing label files ...'), percent, -1)
            num_counter += 1
            self.checkAborted()

    def toLabelFile(self, image_path, samples):
        shapes = []
        for sample in samples:
            shapes.append({
                'label': sample.label,
                'line_color': None,
                'fill_color': None,
                'points': sample.points,
                'shape_type': sample.shape_type,
            })
            self.checkAborted()
        local_image_path = os.path.basename(image_path)
        image_data = LabelFile.load_image_file(image_path)
        label_file_name = os.path.splitext(image_path)[0] + LabelFile.suffix
        label_file = LabelFile()
        label_file.save(
            label_file_name,
            shapes,
            local_image_path,
            sample.image_size[0],
            sample.image_size[1],
            imageData=image_data,
            lineColor=None,
            fillColor=None,
            otherData=None,
            flags=None,
        )
        
    def setIncludedLabels(self, labels):
        self.included_labels = set(labels)

    def setValidationRatio(self, validation_ratio):
        if validation_ratio < 1.0:
            self.validation_ratio = validation_ratio
        else:
            raise Exception('Validation ratio must be >= 0.0 and < 1.0')
    
    def getImages(self):
        return self.images

    def getNumberOfImages(self):
        return len(self.images)

    def getLabels(self):
        return self.labels

    def getNumberOfLabels(self):
        return len(self.labels)
    
    def getSamples(self):
        return self.samples

    def getNumberOfSamples(self):
        return len(self.samples)

    def getSamplesPerLabel(self):
        return self.samplesPerLabel

    def getIncludedLabels(self):
        return self.included_labels

    def getValidationRatio(self):
        return self.validation_ratio

    def getRoot(self):
        return self.root