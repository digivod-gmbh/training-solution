import os
import shutil
import lxml.builder
import lxml.etree
import numpy as np
import PIL.Image

from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config import Export


class FormatVoc(DatasetFormat):

    _files = {
        'class_names': 'class_names.txt',
    }
    _format = 'voc'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.needed_files = [
            FormatVoc._files['class_names'],
        ]
        FormatVoc._files['labels'] = Export.config('labels_file')

    def getTrainFile(self, dataset_path):
        pass

    def getValFile(self, dataset_path):
        pass

    def import_folder(self):
        pass

    def export(self):
        if self.intermediate is None:
            raise Exception('Intermediate format must be initialized for export')
        
        self.thread.update.emit(_('Gathering samples ...'), -1)
        self.checkAborted()

        num_samples = 0
        output_folder = self.output_folder

        os.makedirs(os.path.join(output_folder, 'JPEGImages'))
        os.makedirs(os.path.join(output_folder, 'Annotations'))

        # labels
        labels = self.intermediate.getLabels()
        label_file = os.path.join(output_folder, FormatVoc._files['labels'])
        with open(label_file, 'w+') as f:
            label_txt = '\n'.join(labels)
            f.write(label_txt)
        class_names = list(labels)
        class_names.insert(0, '_background_') # Add background class at beginning
        class_names_file = os.path.join(output_folder, FormatVoc._files['class_names'])
        with open(class_names_file, 'w+') as f:
            label_txt = '\n'.join(class_names)
            f.write(label_txt)

        samples_per_image = self.intermediate.getSamplesPerImage()
        for image in samples_per_image:
            samples = samples_per_image[image]
            num_samples = num_samples + len(samples)
            base = os.path.splitext(image)[0]
            out_img_file = os.path.join(output_folder, 'JPEGImages', base + '.jpg')
            out_xml_file = os.path.join(output_folder, 'Annotations', base + '.xml')
            img_file = os.path.join(self.input_folder, os.path.basename(image))
            img = np.asarray(PIL.Image.open(img_file))
            if not os.path.exists(out_img_file):
                PIL.Image.fromarray(img).save(out_img_file)

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
                        maker.truncated(),
                        maker.difficult(),
                        maker.bndbox(
                            maker.xmin(str(xmin)),
                            maker.ymin(str(ymin)),
                            maker.xmax(str(xmax)),
                            maker.ymax(str(ymax)),
                        ),
                    )
                )

            with open(out_xml_file, 'wb') as f:
                f.write(lxml.etree.tostring(xml, pretty_print=True))

        # save
        config_file = os.path.join(output_folder, Export.config('config_file'))
        files = list(FormatVoc._files.values())
        self.saveConfig(config_file, FormatVoc._format, files, num_samples, self.args)