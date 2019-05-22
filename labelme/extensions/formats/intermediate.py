import os
import random

from labelme.logger import logger
from labelme.label_file import LabelFile


class IntermediateSample():

    def __init__(self, image, image_size, label, points, shape_type):
        self.image = image
        self.image_size = image_size
        self.label = label
        self.points = points
        self.shape_type = shape_type


class IntermediateFormat():
    
    def __init__(self):
        self.root = None
        self.images = set([])
        self.labels = set([])
        self.samples = []
        self.samplesPerLabel = {}
        self.included_labels = set([])
        self.validation_ratio = 0.0

    def addSample(self, image, image_size, label, points, shape_type):
        if len(self.included_labels) > 0 and label not in self.included_labels:
            return
        sample = IntermediateSample(image, image_size, label, points, shape_type)
        self.samples.append(sample)
        self.labels.add(label)
        self.images.add(image)
        if not label in self.samplesPerLabel:
            self.samplesPerLabel[label] = []
        self.samplesPerLabel[label].append(sample)

    def getTrainValidateSamples(self, shuffle=False):
        train_samples = []
        val_samples = []
        for label in self.labels:
            train_for_label, val_for_label = self.getTrainValidateSamplesPerLabel(label, shuffle)
            train_samples += train_for_label
            val_samples += val_for_label
        if shuffle:
            random.shuffle(train_samples)
            random.shuffle(val_samples)
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
        return train_samples, val_samples

    def getLabelFilesFromDataFolder(self, data_folder):
        label_files = []
        for root, dirs, files in os.walk(data_folder):
            for f in files:
                if LabelFile.is_label_file(f):
                    full_path = os.path.normpath(os.path.join(data_folder, f))
                    label_files.append(full_path)
        return label_files

    def addFromLabelFiles(self, data_folder, shuffle=False):
        self.root = data_folder
        label_files = self.getLabelFilesFromDataFolder(data_folder)
        for label_file in label_files:
            self.addFromLabelFile(label_file)
        if shuffle:
            for label_samples in self.samplesPerLabel:
                random.shuffle(self.samplesPerLabel[label_samples])

    def addFromLabelFile(self, label_file):
        lf = LabelFile(label_file)
        image_size = (lf.imageHeight, lf.imageWidth)
        for s in lf.shapes:
            self.addSample(lf.imagePath, image_size, s[0], s[1], s[4])

    def toLabelFiles(self, output_folder):
        # TODO: Implement
        pass

    def toLabelFile(self, output_folder, image):
        # TODO: Implement
        pass
        
    def setIncludedLabels(self, labels):
        self.included_labels = set(labels)

    def setValidationRatio(self, validation_ratio):
        if validation_ratio < 1.0:
            self.validation_ratio = validation_ratio
        else:
            raise Exception('Validation ratio must be >= 0.0 and < 1.0')
    
    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels
    
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