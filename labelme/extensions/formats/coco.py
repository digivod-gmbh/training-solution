import os
import json
import datetime
import collections
import numpy as np
import PIL.Image
import pycocotools.mask

from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config import Export
from labelme.utils import shape_to_mask


class FormatCoco(DatasetFormat):

    _files = {
        'annotations': 'annotations.json',
    }
    _format = 'coco'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.needed_files = [
            FormatCoco._files['annotations'],
        ]
        FormatCoco._files['labels'] = Export.config('labels_file')

    def getTrainFile(self, dataset_path):
        pass

    def getValFile(self, dataset_path):
        pass

    def import_folder(self, folder):
        pass

    def export(self):
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

        out_ann_file = os.path.join(self.output_folder, 'annotations.json')
        image_folder = os.path.join(self.output_folder, 'JPEGImages')
        os.makedirs(image_folder)

        image_id = 0
        samples_per_image = self.intermediate.getSamplesPerImage()
        num_samples = 0

        for image in samples_per_image:
            samples = samples_per_image[image]
            num_samples = num_samples + len(samples)
            base = os.path.splitext(image)[0]
            out_img_file = os.path.join(image_folder, base + '.jpg')
            img_file = os.path.join(self.input_folder, os.path.basename(image))
            img = np.asarray(PIL.Image.open(img_file))
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

        # labels
        label_file = os.path.join(self.output_folder, FormatCoco._files['labels'])
        with open(label_file, 'w+') as f:
            label_txt = '\n'.join(labels)
            f.write(label_txt)

        # save
        config_file = os.path.join(self.output_folder, Export.config('config_file'))
        files = list(FormatCoco._files.values())
        num_samples = {
            'train': num_samples,
            # TODO: Add validation dataset
        }
        self.saveConfig(config_file, FormatCoco._format, files, num_samples, self.args)

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


