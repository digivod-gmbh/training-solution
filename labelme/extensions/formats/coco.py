import os
import json
import shutil
import datetime
import collections
import numpy as np
import PIL.Image
import pycocotools.mask

from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config import Export
from labelme.logger import logger
from labelme.utils import shape_to_mask


class FormatCoco(DatasetFormat):

    _files = {
        'annotations_train': 'annotations_train.json',
        'annotations_val': 'annotations_val.json',
    }
    _format = 'coco'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.needed_files = [
            FormatCoco._files['annotations_train'],
            #FormatCoco._files['annotations_val'],
        ]
        FormatCoco._files['labels'] = Export.config('labels_file')

    def getTrainFile(self, dataset_path):
        pass

    def getValFile(self, dataset_path):
        pass

    def import_folder(self):
        if self.input_folder is None:
            raise Exception('Input folder must be initialized for import')

        if not self.args.config['format'] == FormatCoco._format:
            raise Exception('Format {} in config file does not match {}'.format(self.args.config.format, FormatCoco._format))

        input_folder = self.input_folder
        output_folder = self.output_folder

        self.intermediate = IntermediateFormat()

        annotations_train = os.path.join(input_folder, FormatCoco._files['annotations_train'])
        self.import_to_intermediate(annotations_train, output_folder, input_folder)

        if self.args.config['args']['validation_ratio'] > 0.0:
            annotations_val = os.path.join(input_folder, FormatCoco._files['annotations_val'])
            self.import_to_intermediate(annotations_val, output_folder, input_folder)

        self.intermediate.toLabelFiles()

    def import_to_intermediate(self, annotation_file, output_folder, input_folder):
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        class_id_to_name = {}
        for category in data['categories']:
            class_id_to_name[category['id']] = category['name']
        
        image_id_to_path = {}
        image_id_to_size = {}
        for image in data['images']:
            src_image = os.path.join(input_folder, image['file_name'])
            dst_image = os.path.join(output_folder, os.path.basename(image['file_name']))
            if not os.path.exists(dst_image):
                shutil.copyfile(src_image, dst_image)
            image_id_to_path[image['id']] = dst_image
            image_id_to_size[image['id']] = (image['height'], image['width'])   

        for annotation in data['annotations']:
            image_id = annotation['image_id']
            image_path = image_id_to_path[image_id]
            image_size = image_id_to_size[image_id]
            label_name = class_id_to_name[annotation['category_id']]
            segmentations = annotation['segmentation']
            for bbox in segmentations:
                points = [
                    [int(bbox[0]), int(bbox[1])],
                    [int(bbox[2]), int(bbox[3])],
                ]
                self.intermediate.addSample(image_path, image_size, label_name, points, 'rectangle')

    def export(self):
        samples_per_image = {}
        num_samples_val = 0
        validation_ratio = self.intermediate.getValidationRatio()
        if validation_ratio > 0.0:
            train_samples, val_samples = self.intermediate.getTrainValidateSamples(shuffle=True)
            samples_per_image_train = self.intermediate.getSamplesPerImage(train_samples)
            num_samples_train = self.saveDataset(samples_per_image_train, FormatCoco._files['annotations_train'])
            samples_per_image_val = self.intermediate.getSamplesPerImage(val_samples)
            num_samples_val = self.saveDataset(samples_per_image_val, FormatCoco._files['annotations_val'])
        else:
            samples_per_image = self.intermediate.getSamplesPerImage()
            num_samples_train = self.saveDataset(samples_per_image, FormatCoco._files['annotations_train'])

        # labels
        labels = self.intermediate.getLabels()
        label_file = os.path.join(self.output_folder, FormatCoco._files['labels'])
        with open(label_file, 'w+') as f:
            label_txt = '\n'.join(labels)
            f.write(label_txt)

        # save
        config_file = os.path.join(self.output_folder, Export.config('config_file'))
        files = list(FormatCoco._files.values())
        if validation_ratio <= 0.0:
            # remove val files from file list for config
            val_files = [FormatCoco._files['annotations_val']]
            files = [x for x in files if x not in val_files]
        num_samples = {
            'train': num_samples_train,
            'val': num_samples_val,
        }
        self.saveConfig(config_file, FormatCoco._format, files, num_samples, self.args)

    def saveDataset(self, samples_per_image, file_name):
        image_id = 0
        num_samples = 0

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
        out_ann_file = os.path.join(output_folder, file_name)
        image_folder = os.path.join(output_folder, 'JPEGImages')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        for image in samples_per_image:
            samples = samples_per_image[image]
            num_samples = num_samples + len(samples)
            base = os.path.splitext(image)[0]
            out_img_file = os.path.join(image_folder, base + '.jpg')
            img_file = os.path.join(self.input_folder, os.path.basename(image))
            img = np.asarray(PIL.Image.open(img_file))
            if not os.path.exists(out_img_file):
                PIL.Image.fromarray(img).save(out_img_file)

            data['images'].append(dict(
                license=0,
                url=None,
                file_name=os.path.relpath(out_img_file, os.path.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            ))

            masks = {}                                     # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for sample in samples:
                points = sample.points
                label = sample.label
                shape_type = sample.shape_type
                mask = shape_to_mask(
                    img.shape[:2], points, shape_type
                )

                if label in masks:
                    masks[label] = masks[label] | mask
                else:
                    masks[label] = mask

                points = np.asarray(points).flatten().tolist()
                segmentations[label].append(points)

            for label, mask in masks.items():
                cls_name = label.split('-')[0]
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data['annotations'].append(dict(
                    id=len(data['annotations']),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[label],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                ))

            image_id = image_id + 1

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


