import os
import glob
import shutil
import lxml.builder
import lxml.etree
import numpy as np
import PIL.Image
from gluoncv import data

from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config.export import Export
from labelme.logger import logger


class FormatVoc(DatasetFormat):

    _files = {
        'all_splits': 'all.txt',
    }
    _splits = {
        'train': 'train',
        'val': 'val',
    }
    _directories = {
        'annotations': 'Annotations',
        'images': 'JPEGImages',
        'splits': 'ImageSets/Main',
    }
    _format = 'voc'

    def __init__(self, thread = None, all_image_sets = False):
        super().__init__()
        self.thread = thread
        self.intermediate = None
        self.dataset = None
        self.all_image_sets = all_image_sets
        FormatVoc._files['labels'] = Export.config('labels_file')

    def isValidFormat(self, dataset_folder_or_file):
        root_folder = dataset_folder_or_file
        if self.all_image_sets:
            if not os.path.isdir(dataset_folder_or_file):
                logger.warning('Dataset folder {} does not exist'.format(dataset_folder_or_file))
                return False
        else:
            root_folder = self._getRootFolderFromFile(dataset_folder_or_file)
            if not os.path.isfile(dataset_folder_or_file):
                logger.warning('Dataset file {} does not exist'.format(dataset_folder_or_file))
                return False
        annotations_dir = os.path.join(root_folder, FormatVoc._directories['annotations'])
        if not os.path.isdir(annotations_dir):
            logger.warning('Annotations folder {} does not exist'.format(annotations_dir))
            return False
        images_dir = os.path.join(root_folder, FormatVoc._directories['images'])
        if not os.path.isdir(images_dir):
            logger.warning('Images folder {} does not exist'.format(images_dir))
            return False
        return True

    def _getRootFolderFromFile(self, split_file):
        return os.path.normpath(os.path.join(os.path.dirname(split_file), '../../'))

    def getLabels(self):
        dataset = self._loadDataset()
        return dataset.classes

    def getNumSamples(self):
        logger.debug('Count samples in dataset')
        dataset = self._loadDataset()
        return len(dataset)

    def getDatasetForTraining(self):
        dataset = self._loadDataset()
        return dataset

    def _loadDataset(self):
        if self.dataset is None:
            root_folder = self.input_folder_or_file
            if self.all_image_sets:
                self.dataset = VOCDetectionCustom(self.input_folder_or_file, splits=FormatVoc._splits['train'].values(), transform=None, index_map=None, preload_label=True)
            else:
                root_folder = self._getRootFolderFromFile(self.input_folder_or_file)
                split_file = os.path.basename(self.input_folder_or_file)
                splits = [os.path.splitext(split_file)[0]]
                self.dataset = VOCDetectionCustom(root_folder, splits=splits, transform=None, index_map=None, preload_label=True)
        return self.dataset

    def importFolder(self):
        if self.input_folder_or_file is None:
            raise Exception('Input folder must be initialized for import')

        self.intermediate = IntermediateFormat()
        self.importToIntermediate(self.output_folder, self.input_folder_or_file)
        self.intermediate.toLabelFiles()

    def importToIntermediate(self, output_folder, input_folder_or_file):
        input_folder = input_folder_or_file
        if not os.path.isdir(input_folder):
            input_folder = self._getRootFolderFromFile(input_folder_or_file)

        annotations_dir = os.path.join(input_folder, FormatVoc._directories['annotations'])
        annotation_files = []
        if self.all_image_sets:
            annotation_files = os.listdir(annotations_dir)
        else:
            with open(input_folder_or_file, 'r') as f:
                lines = [l.strip().split() for l in f.readlines()]
                for l in lines:
                    try:
                        num = int(l[1])
                        if num > -1:
                            annotation_files.append(str(l[0] + '.xml'))
                    except:
                        pass

        self.thread.update.emit(_('Loading dataset ...'), 10, -1)
        self.checkAborted()

        for idx1, annotation_file in enumerate(annotation_files):
            file_path = os.path.join(annotations_dir, annotation_file)
            root = lxml.etree.parse(file_path)
            filename = root.find('filename').text

            self.checkAborted()

            # Copy image
            src_image = os.path.join(input_folder, FormatVoc._directories['images'], filename)
            dst_image = os.path.join(output_folder, filename)
            if not os.path.exists(dst_image):
                shutil.copyfile(src_image, dst_image)

            # Additional elements
            #folder = root.find('folder').text
            #database = root.find('database').text
            #annotation = root.find('annotation').text
            #image = root.find('image').text
            #segmented = root.find('segmented').text
            
            size = root.find('size')
            image_size = (int(size.find('height').text), int(size.find('width').text))
            
            self.checkAborted()

            objects = root.findall('object')
            for idx2, obj in enumerate(objects):
                bbox = obj.find('bndbox')
                points = [
                    [int(bbox.find('xmin').text), int(bbox.find('ymin').text)],
                    [int(bbox.find('xmax').text), int(bbox.find('ymax').text)],
                ]
                label_name = obj.find('name').text
                self.intermediate.addSample(dst_image, image_size, label_name, points, 'rectangle')

                percentage = idx2 / len(objects) * 90 * idx1 / len(annotation_files)
                self.thread.update.emit(_('Loading dataset ...'), 10 + percentage, -1)
                self.checkAborted()

    def export(self):
        if self.intermediate is None:
            raise Exception('Intermediate format must be initialized for export')
        
        self.thread.update.emit(_('Gathering samples ...'), -1, -1)
        self.checkAborted()

        num_samples = 0
        input_folder = self.input_folder_or_file
        output_folder = self.output_folder

        os.makedirs(os.path.join(output_folder, FormatVoc._directories['images']))
        os.makedirs(os.path.join(output_folder, FormatVoc._directories['annotations']))
        os.makedirs(os.path.join(output_folder, FormatVoc._directories['splits']))

        samples_per_image = {}
        validation_ratio = self.intermediate.getValidationRatio()
        if validation_ratio > 0.0:
            train_samples, val_samples = self.intermediate.getTrainValidateSamples(shuffle=True, group_by_images=True)
            samples_per_image_train = self.intermediate.getSamplesPerImage(train_samples)
            num_samples_train = self.saveDataset(samples_per_image_train, output_folder, input_folder, FormatVoc._splits['train'])
            samples_per_image_val = self.intermediate.getSamplesPerImage(val_samples)
            num_samples_val = self.saveDataset(samples_per_image_val, output_folder, input_folder, FormatVoc._splits['val'])
        else:
            samples_per_image = self.intermediate.getSamplesPerImage()
            num_samples_train = self.saveDataset(samples_per_image, output_folder, input_folder, FormatVoc._splits['train'])

        all_split_file = os.path.join(output_folder, FormatVoc._directories['splits'], FormatVoc._files['all_splits'])
        open(all_split_file, 'w').close()
        with open(all_split_file, 'a') as f:
            f.write('\n'.join([os.path.splitext(image)[0] for image in self.intermediate.getImages()]))

        # labels
        labels = self.intermediate.getLabels()
        label_file = os.path.join(output_folder, FormatVoc._files['labels'])
        with open(label_file, 'w+') as f:
            label_txt = '\n'.join(labels)
            f.write(label_txt)

    def saveDataset(self, samples_per_image, output_folder, input_folder, split):
        num_samples = 0
        labels = self.intermediate.getLabels()
        class_names = list(labels)

        self.checkAborted()

        out_split_file = os.path.join(output_folder, FormatVoc._directories['splits'], split + '.txt')
        open(out_split_file, 'w').close()

        for image in samples_per_image:
            samples = samples_per_image[image]
            num_samples = num_samples + len(samples)
            base = os.path.splitext(image)[0]
            out_img_file = os.path.join(output_folder, FormatVoc._directories['images'], base + '.jpg')
            out_xml_file = os.path.join(output_folder, FormatVoc._directories['annotations'], base + '.xml')
            img_file = os.path.join(input_folder, os.path.basename(image))
            img = np.asarray(PIL.Image.open(img_file))
            if not os.path.exists(out_img_file):
                PIL.Image.fromarray(img).save(out_img_file)

            self.checkAborted()

            samples_count = len(samples) if len(samples) > 0 else -1
            with open(out_split_file, 'a') as f:
                f.write(base + ' ' + str(samples_count) + '\n')

            maker = lxml.builder.ElementMaker()
            xml = maker.annotation(
                maker.folder(),
                maker.filename(base + '.jpg'),
                maker.database(),    # e.g., The VOC2007 Database
                maker.annotation(),  # e.g., Pascal VOC2007
                maker.image(),       # e.g., flickr
                maker.size(
                    maker.height(str(img.shape[0])),
                    maker.width(str(img.shape[1])),
                    maker.depth(str(img.shape[2])),
                ),
                maker.segmented(),
            )

            self.checkAborted()

            bboxes = []
            labels = []
            for sample in samples:
                points = sample.points
                label = sample.label
                shape_type = sample.shape_type

                if shape_type != 'rectangle':
                    continue

                class_name = label
                class_id = class_names.index(class_name)

                (xmin, ymin), (xmax, ymax) = points
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

                xml.append(
                    maker.object(
                        maker.name(label),
                        maker.pose(),
                        maker.truncated(str(0)),
                        maker.difficult(str(0)),
                        maker.bndbox(
                            maker.xmin(str(xmin)),
                            maker.ymin(str(ymin)),
                            maker.xmax(str(xmax)),
                            maker.ymax(str(ymax)),
                        ),
                    )
                )
                
                self.thread.update.emit(_('Writing sample ...'), -1, -1)
                self.checkAborted()

            self.checkAborted()

            with open(out_xml_file, 'wb') as f:
                f.write(lxml.etree.tostring(xml, pretty_print=True))

        return num_samples


class VOCDetectionCustom(data.VOCDetection):
    """ Custom class for VOCDetection to avoid pre-defined classes and splits """

    def __init__(self, root, splits=[FormatVoc._splits['train']], transform=None, index_map=None, preload_label=True):
        self._classes = None
        super(VOCDetectionCustom, self).__init__(root, splits, transform, index_map, preload_label)

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        if self._classes is None:
            try:
                label_file = os.path.join(self._root, Export.config('labels_file'))
                with open(label_file, 'r') as f:
                    labels = [l.strip().lower() for l in f.readlines()]
                self._validate_class_names(labels)
                self._classes = labels
            except AssertionError as e:
                raise RuntimeError("Class names must not contain {}".format(e))
        return self._classes

    def _load_items(self, splits):
        ids = []
        for name in splits:
            lf = os.path.join(self._root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(self._root, line.strip()) for line in f.readlines()]
        return ids
