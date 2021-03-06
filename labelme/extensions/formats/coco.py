import os
import json
import shutil
import datetime
import traceback
import collections
import numpy as np
import PIL.Image
import pycocotools.mask
from gluoncv import data

from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config.export import Export
from labelme.logger import logger
from labelme.utils import shape_to_mask, save_image_as_jpeg, replace_special_chars
from labelme.config import MessageType


class FormatCoco(DatasetFormat):

    _files = {
        'instances': 'instances',
    }
    _directories = {
        'annotations': 'annotations',
    }
    _splits = {
        'train': 'train',
        'val': 'val',
    }
    _format = 'coco'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.dataset = None
        self.num_samples = -1
        FormatCoco._files['labels'] = Export.config('labels_file')

    def getOutputFileName(self, split='train'):
        if split in FormatCoco._splits:
            file_name = '{}_{}.json'.format(FormatCoco._files['instances'], FormatCoco._splits[split])
            return os.path.join(FormatCoco._directories['annotations'], file_name)
        raise Exception('Unknown split {}'.format(split))

    def isValidFormat(self, dataset_folder_or_file):
        if not os.path.isfile(dataset_folder_or_file):
            logger.warning('Dataset file {} does not exist'.format(dataset_folder_or_file))
            return False
        try:
            with open(dataset_folder_or_file, 'r') as f:
                data = json.load(f)
            return True
        except Exception as e:
            logger.warning('Error during parsing of json file {}: {}'.format(dataset_folder_or_file, e))
            return False

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
            root_folder = os.path.normpath(os.path.dirname(self.input_folder_or_file) + '/../')
            split = self.fileNameToSplit(self.input_folder_or_file)
            self.dataset = COCODetectionCustom(root_folder, splits=[split])
        return self.dataset

    def importFolder(self):
        if self.input_folder_or_file is None:
            raise Exception(_('Input folder must be initialized for import'))

        input_folder = os.path.dirname(self.input_folder_or_file)
        output_folder = self.output_folder

        self.intermediate = IntermediateFormat()
        self.intermediate.setAbortable(self.abortable)
        self.intermediate.setThread(self.thread)
        self.importToIntermediate(self.input_folder_or_file, output_folder, input_folder)
        self.thread.update.emit(_('Writing label files ...'), 90, -1)
        self.intermediate.toLabelFiles()

    def fileNameToSplit(self, file_name):
        split_base = os.path.splitext(os.path.basename(file_name))[0]
        pos = split_base.find('_')
        if pos > -1:
            return split_base[pos+1:]
        return ''

    def importToIntermediate(self, annotation_file, output_folder, input_folder):
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        class_id_to_name = {}
        for category in data['categories']:
            class_id_to_name[category['id']] = category['name']
        
        split = self.fileNameToSplit(annotation_file)
        if split is False:
            raise Exception(_('Split could not be detected'))
        else:
            split = '../' + split

        self.thread.update.emit(_('Loading dataset ...'), 10, -1)
        self.checkAborted()

        image_id_to_path = {}
        image_id_to_size = {}
        for idx, image in enumerate(data['images']):
            src_image = os.path.join(input_folder, split, image['file_name'])
            dst_image = os.path.join(output_folder, os.path.basename(image['file_name']))
            if not os.path.exists(dst_image):
                shutil.copyfile(src_image, dst_image)
            image_id_to_path[image['id']] = dst_image
            image_id_to_size[image['id']] = (image['height'], image['width'])

            percentage = idx / len(data['images']) * 40
            self.thread.update.emit(_('Loading dataset ...'), 10 + percentage, -1)
            self.checkAborted()

        for idx, annotation in enumerate(data['annotations']):
            image_id = annotation['image_id']
            image_path = image_id_to_path[image_id]
            image_size = image_id_to_size[image_id]
            label_name = class_id_to_name[annotation['category_id']]
            segmentations = annotation['segmentation']
            # In case segmentation is in form [[x1, y1, x2, y2, ...]]
            if len(segmentations) == 1 and len(segmentations[0]) > 1:
                segmentations = annotation['segmentation'][0]
            is_crowd = annotation['iscrowd'] == 1
            points = []

            try:
                # Skip annotations with flag iscrowd
                if not is_crowd:
                    for i in range(0, len(segmentations)-1, 2):
                        points.append([float(segmentations[i]), float(segmentations[i+1])])
                    if self.isPolygon(points):
                        if self.isBBox(points):
                            points = self.bboxToPolygon(points)
                        self.intermediate.addSample(image_path, image_size, label_name, points, 'polygon')
                    else:
                        bbox = annotation['bbox']
                        points = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                        self.intermediate.addSample(image_path, image_size, label_name, points, 'rectangle')
            except Exception as e:
                logger.error(traceback.format_exc())

            percentage = idx / len(data['annotations']) * 40
            self.thread.update.emit(_('Loading dataset ...'), 50 + percentage, -1)
            self.checkAborted()

    def export(self):
        if self.intermediate is None:
            raise Exception(_('Intermediate format must be initialized for export'))
        
        self.thread.update.emit(_('Gathering samples ...'), -1, -1)
        self.checkAborted()
        
        samples_per_image = {}
        num_samples_val = 0
        validation_ratio = self.intermediate.getValidationRatio()
        if validation_ratio > 0.0:
            train_samples, val_samples = self.intermediate.getTrainValidateSamples(shuffle=True)
            samples_per_image_train = self.intermediate.getSamplesPerImage(train_samples)
            num_samples_train = self.saveDataset(samples_per_image_train, FormatCoco._splits['train'])
            samples_per_image_val = self.intermediate.getSamplesPerImage(val_samples)
            num_samples_val = self.saveDataset(samples_per_image_val, FormatCoco._splits['val'])
        else:
            samples_per_image = self.intermediate.getSamplesPerImage()
            num_samples_train = self.saveDataset(samples_per_image, FormatCoco._splits['train'])

        # labels
        labels = self.intermediate.getLabels()
        label_file = os.path.join(self.output_folder, FormatCoco._files['labels'])
        with open(label_file, 'w+') as f:
            label_txt = '\n'.join(labels)
            f.write(label_txt)

    def saveDataset(self, samples_per_image, split):
        image_id = 0
        num_samples = 0

        self.checkAborted()

        data = self.getEmptyData()

        class_name_to_id = {}
        labels = self.intermediate.getLabels()
        for i, label in enumerate(labels):
            class_id = i
            class_name = label.strip()
            class_name_to_id[class_name] = class_id
            data['categories'].append(dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            ))

        output_folder = self.output_folder
        output_file = self.getOutputFileName(split)
        out_ann_file = os.path.join(output_folder, output_file)
        out_ann_dir = os.path.dirname(out_ann_file)
        if not os.path.exists(out_ann_dir):
            os.makedirs(out_ann_dir)
        image_folder = os.path.join(output_folder, split)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        input_folder = self.input_folder_or_file

        self.checkAborted()

        failed_images = []
        for image in samples_per_image:
            try:
                samples = samples_per_image[image]
                num_samples = num_samples + len(samples)
                base = replace_special_chars(os.path.splitext(image)[0])
                out_img_file = os.path.join(image_folder, base + '.jpg')
                img_file = os.path.normpath(os.path.join(input_folder, image))
                image = PIL.Image.open(img_file)
                img = np.asarray(image)
                if not os.path.exists(out_img_file):
                    save_image_as_jpeg(image, out_img_file)

                self.checkAborted()

                data['images'].append(dict(
                    license=0,
                    url=None,
                    file_name=os.path.basename(out_img_file),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                ))

                masks = {}                                     # for area
                segmentations = collections.defaultdict(list)  # for segmentation
                for sample in samples:
                    label = sample.label
                    shape_type = sample.shape_type
                    points = sample.points
                    
                    mask = shape_to_mask(
                        img.shape[:2], points, shape_type
                    )

                    points = np.asarray(points).flatten().tolist()

                    self.checkAborted()

                    if label in masks:
                        masks[label].append(mask)
                        segmentations[label].append(points)
                    else:
                        masks[label] = [mask]
                        segmentations[label] = [points]

                for label, mask in masks.items():
                    for i in range(len(mask)):
                        m = mask[i]
                        cls_name = label.split('-')[0]
                        if cls_name not in class_name_to_id:
                            continue
                        cls_id = class_name_to_id[cls_name]

                        m = np.asfortranarray(m.astype(np.uint8))
                        m = pycocotools.mask.encode(m)
                        area = float(pycocotools.mask.area(m))
                        bbox = pycocotools.mask.toBbox(m).flatten().tolist()

                        data['annotations'].append(dict(
                            id=len(data['annotations']),
                            image_id=image_id,
                            category_id=cls_id,
                            segmentation=segmentations[label][i],
                            area=area,
                            bbox=bbox,
                            iscrowd=0,
                        ))

                        self.thread.update.emit(_('Writing samples ...'), -1, -1)
                        self.checkAborted()

                image_id = image_id + 1

            except Exception as e:
                failed_images.append(image)
                logger.error(traceback.format_exc())

        if len(failed_images) > 0:
            msg = _('The following images could not be exported:') + '\n' + ', '.join(failed_images)
            self.thread.message.emit(_('Warning'), msg, MessageType.Warning)
            if len(data['annotations']) == 0:
                self.throwUserException(_('Dataset contains no images for export'))

        self.checkAborted()

        with open(out_ann_file, 'w') as f:
            json.dump(data, f)

        return num_samples

    def getEmptyData(self):
        now = datetime.datetime.now()
        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
            ),
            licenses=[dict(
                url=None,
                id=0,
                name=None,
            )],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type='instances',
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )
        return data

    def isPolygon(self, points):
        if len(points) == 4:
            # Rectangle
            isRectangle = True
            for i in range(4):
                # Do x and y coordinates match the next or previous point? Yes > rectangle, No > polygon
                if (not (points[i][0] == points[(i+1)%4][0] or points[i][0] == points[(i+3)%4][0] and   # X
                    points[i][1] == points[(i+1)%4][1] or points[i][1] == points[(i+3)%4][1])):         # Y
                    isRectangle = False
                    break
            return not isRectangle
        # Polygon
        return True

    def isBBox(self, bbox):
        return len(bbox) == 2 and len(bbox[0]) == 2 and len(bbox[1]) == 2

    def bboxToPolygon(self, bbox):
        # [[xmin, ymin], [xmax, ymax]]
        if not self.isBBox(bbox):
            raise Exception(_('Invalid bbox list: {}').format(bbox))
        xmin = bbox[0][0]
        ymin = bbox[0][1]
        xmax = bbox[1][0]
        ymax = bbox[1][1]
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

class COCODetectionCustom(data.COCODetection):
    """ Custom class for COCODetection to avoid pre-defined classes and splits """

    def __init__(self, root, splits=[FormatCoco._splits['train']], transform=None, min_object_area=0, skip_empty=True, use_crowd=True):
        self._classes = None
        # for i, split in enumerate(splits):
        #     splits[i] = FormatCoco._files['instances'] + '_' + split
        super(COCODetectionCustom, self).__init__(root, splits, transform, min_object_area, skip_empty, use_crowd)

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    def _parse_image_path(self, entry):
        abs_path = os.path.join(self._root, )
        dirname, filename = entry['coco_url'].split('/')[-2:]
        abs_path = os.path.join(self._root, dirname, filename)
        return abs_path

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        im_aspect_ratios = []
        from pycocotools.coco import COCO
        for split in self._splits:
            file_name = FormatCoco._files['instances'] + '_' + split + '.json'
            anno = os.path.join(self._root, self.annotation_dir, file_name) 
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                abs_path = os.path.join(self._root, split, entry['file_name'])
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_bbox(_coco, entry)
                if not label:
                    continue
                im_aspect_ratios.append(float(entry['width']) / entry['height'])
                items.append(abs_path)
                labels.append(label)
        return items, labels

    @property
    def classes(self):
        if self._classes is None:
            try:
                label_file = os.path.join(self._root, Export.config('labels_file'))
                with open(label_file, 'r') as f:
                    labels = [l.strip() for l in f.readlines()]
                self._classes = labels
            except AssertionError as e:
                raise RuntimeError("Class names must not contain {}".format(e))
        return self._classes