import numpy as np
import os
import sys
import itertools

import mxnet as mx
import random
from PIL import Image
import time
import traceback
import builtins
import cv2

from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme.utils.map import Map
from labelme.config import Export

from .format import DatasetFormat
from .intermediate import IntermediateFormat


class FormatImageRecord(DatasetFormat):

    _files = {
        'lst_train': 'train.lst', 
        'lst_val': 'val.lst', 
        'rec_train': 'train.rec', 
        'rec_val': 'val.rec',
        'idx_train': 'train.idx', 
        'idx_val': 'val.idx',
    }
    _format = 'imagerecord'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.needed_files = [
            FormatImageRecord._files['rec_train'],
            FormatImageRecord._files['idx_train'],
            #FormatImageRecord._files['lst_train'],
        ]
        FormatImageRecord._files['labels'] = Export.config('labels_file')

    def getTrainFile(self, dataset_path):
        train_file = os.path.join(dataset_path, FormatImageRecord._files['rec_train'])
        return train_file

    def getValFile(self, dataset_path):
        val_file = os.path.join(dataset_path, FormatImageRecord._files['rec_val'])
        return val_file

    def import_folder(self):
        if self.input_folder is None:
            raise Exception('Input folder must be initialized for import')

        if not self.args.config['format'] == FormatImageRecord._format:
            raise Exception('Format {} in config file does not match {}'.format(self.args.config.format, FormatImageRecord._format))

        input_folder = self.input_folder
        output_folder = self.output_folder

        self.intermediate = IntermediateFormat()

        train_rec_file = os.path.join(input_folder, FormatImageRecord._files['rec_train'])
        self.import_to_intermediate(train_rec_file, output_folder)

        if self.args.config['args']['validation_ratio'] > 0.0:
            val_rec_file = os.path.join(input_folder, FormatImageRecord._files['rec_val'])
            self.import_to_intermediate(val_rec_file, output_folder)

        self.intermediate.toLabelFiles()

    def import_to_intermediate(self, rec_file, output_folder):
        all_labels = []
        for i, line in enumerate(open(self.args.label_file).readlines()):
            all_labels.append(line)

        record = mx.recordio.MXRecordIO(rec_file, 'r')
        record.reset()
        while True:
            try:
                item = record.read()
                if not item:
                    break
                header, image = mx.recordio.unpack_img(item)
                img_file = os.path.join(output_folder, '{:09d}.jpg'.format(header.id))
                cv2.imwrite(img_file, image)
                image_height = image.shape[0]
                image_width = image.shape[1]
                shapes = []
                for i in range(4, len(header.label), 5):
                    label_idx = int(header.label[i])
                    bbox = header.label[i+1:i+5] 
                    label_name = _('unknown')
                    if label_idx < len(all_labels):
                        label_name = all_labels[label_idx].strip()
                    points = [
                        [int(bbox[0] * image_width), int(bbox[1] * image_height)],
                        [int(bbox[2] * image_width), int(bbox[3] * image_height)],
                    ]
                    # imagerecord has only rectangle shapes
                    self.intermediate.addSample(img_file, (image_height, image_width), label_name, points, 'rectangle')

            except Exception as e:
                logger.error(e)
                
        record.close()

    def export(self):
        if self.intermediate is None:
            raise Exception('Intermediate format must be initialized for export')
        
        self.thread.update.emit(_('Gathering samples ...'), -1)
        self.checkAborted()

        labels = self.intermediate.getLabels()
        self.label_dict = {}
        for idx, label in enumerate(labels):
            self.label_dict[label] = idx
        train_samples, val_samples = self.intermediate.getTrainValidateSamples(shuffle=True)
        num_train_samples = len(train_samples)
        num_val_samples = len(val_samples)

        output_folder = self.output_folder
        data_folder = self.intermediate.getRoot()

        # labels
        label_file = os.path.join(output_folder, FormatImageRecord._files['labels'])
        with open(label_file, 'w+') as f:
            label_txt = '\n'.join(labels)
            f.write(label_txt)

        # train
        self.thread.update.emit(_('Creating training dataset ...'), -1)
        self.checkAborted()
        self.makeLstFile(output_folder, 'lst_train', train_samples)
        self.makeRecFile(data_folder, output_folder, 'rec_train', 'idx_train', 'lst_train')

        config_file = os.path.join(output_folder, Export.config('config_file'))
        files = list(FormatImageRecord._files.values())

        # validate
        validation_ratio = self.intermediate.getValidationRatio()
        if validation_ratio > 0.0:
            self.thread.update.emit(_('Creating validation dataset ...'), -1)
            self.checkAborted()
            self.makeLstFile(output_folder, 'lst_val', val_samples)
            self.makeRecFile(data_folder, output_folder, 'rec_val', 'idx_val', 'lst_val')
        else:
            # remove val files from file list for config
            val_files = [FormatImageRecord._files['rec_val'], FormatImageRecord._files['idx_val'], FormatImageRecord._files['lst_val']]
            files = [x for x in files if x not in val_files]

        # save
        num_samples = {
            'train': num_train_samples,
            'val': num_val_samples,
        }
        self.saveConfig(config_file, FormatImageRecord._format, files, num_samples, self.args)

    def makeLstFile(self, output_folder, file_key, samples):
        file_name = FormatImageRecord._files[file_key]
        lst_file = os.path.join(output_folder, file_name)

        # group samples by image
        samples_per_image = self.intermediate.getSamplesPerImage()

        with open(lst_file, 'w+') as f:
            idx = 0
            for image in samples_per_image:
                ids = []
                boxes = []
                for sample in samples_per_image[image]:
                    points = self.convertPointsToBoundingBox(sample.points, sample.shape_type)
                    box = []
                    for point in points:
                        box += point
                    boxes.append(box)
                    ids.append(self.label_dict[sample.label])
                ids = np.array(ids, dtype=np.int32)
                line = self.createLstLine(sample.image, sample.image_size, boxes, ids, idx)
                f.write(line)
                idx += 1

    def makeRecFile(self, data_folder, output_folder, rec_file_key, idx_file_key, lst_file_key):
        file_name_rec = FormatImageRecord._files[rec_file_key]
        file_name_idx = FormatImageRecord._files[idx_file_key]
        file_name_lst = FormatImageRecord._files[lst_file_key]

        image_list = self.readLstFile(os.path.join(output_folder, file_name_lst))
        record = mx.recordio.MXIndexedRecordIO(os.path.join(output_folder, file_name_idx), os.path.join(output_folder, file_name_rec), 'w')

        args = Map({
            'root': data_folder,
            'pass_through': True,
            'resize': 0,
            'center_crop': False,
            'quality': 95,
            'encoding': '.jpg',
            'pack_label': True,
        })
        try:
            import Queue as queue
        except ImportError:
            import queue
        q_out = queue.Queue()
        cnt = 0
        pre_time = time.time()
        for i, item in enumerate(image_list):
            self.imageEncode(args, i, item, q_out)
            if q_out.empty():
                continue
            _a, s, _b = q_out.get()
            record.write_idx(item[0], s)
            if cnt % 1000 == 0:
                cur_time = time.time()
                logger.debug('time: {} count: {}'.format(cur_time - pre_time, cnt))
                pre_time = cur_time
            cnt += 1
            self.thread.update.emit(_('Writing dataset ...'), -1)

    def readLstFile(self, path_in):
        """Reads the .lst file and generates corresponding iterator.
        Parameters
        ----------
        path_in: string
        Returns
        -------
        item iterator that contains information in .lst file
        """
        with open(path_in) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                line = [i.strip() for i in line.strip().split('\t')]
                line_len = len(line)
                # check the data format of .lst file
                if line_len < 3:
                    logger.warning('lst should have at least has three parts, but only has {} parts for {}}'.format(line_len, line))
                    continue
                try:
                    item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
                except Exception as e:
                    logger.error('Parsing lst met error for {}, detail: {}'.format(line, e))
                    continue
                yield item

    def imageEncode(self, args, i, item, q_out):
        """Reads, preprocesses, packs the image and puts it back in output queue.
        Parameters
        ----------
        args: object
        i: int
        item: list
        q_out: queue
        """
        fullpath = os.path.join(args.root, item[1])

        if len(item) > 3 and args.pack_label:
            header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
        else:
            header = mx.recordio.IRHeader(0, item[2], item[0], 0)

        if args.pass_through:
            try:
                with open(fullpath, 'rb') as fin:
                    img = fin.read()
                s = mx.recordio.pack(header, img)
                q_out.put((i, s, item))
            except Exception as e:
                traceback.print_exc()
                logger.error('pack_img error:', item[1], e)
                q_out.put((i, None, item))
            return

        try:
            img = Image.open(fullpath)
        except:
            traceback.print_exc()
            logger.error('imread error trying to load file: {}'.format(fullpath))
            q_out.put((i, None, item))
            return
        if img is None:
            logger.error('imread read blank (None) image for file: {}'.format(fullpath))
            q_out.put((i, None, item))
            return
        if args.center_crop:
            if img.shape[0] > img.shape[1]:
                margin = (img.shape[0] - img.shape[1]) // 2
                img = img[margin:margin + img.shape[1], :]
            else:
                margin = (img.shape[1] - img.shape[0]) // 2
                img = img[:, margin:margin + img.shape[0]]
        if args.resize:
            if img.shape[0] > img.shape[1]:
                newsize = (args.resize, img.shape[0] * args.resize // img.shape[1])
            else:
                newsize = (img.shape[1] * args.resize // img.shape[0], args.resize)
            img = img.thumbnail(newsize, Image.ANTIALIAS)

        try:
            s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            logger.error('pack_img error on file: {} {}'.format(fullpath, e))
            q_out.put((i, None, item))
            return

    # https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html#recordfiledetection-for-entire-dataset-packed-in-single-mxnet-recordfile
    def createLstLine(self, img_path, im_shape, boxes, ids, idx):
        h, w = im_shape
        # for header, we use minimal length 2, plus width and height with A: 4, B: 5, C: width, D: height
        A, B, C, D = 4, 5, w, h
        labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
        labels[:, (1, 3)] /= float(w)
        labels[:, (2, 4)] /= float(h)
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = []
        for l in labels:
            str_labels += [str(int(l[0]))] + [str(x) for x in l[1:]]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line

    def convertPointsToBoundingBox(self, points, shape_type):
        if shape_type == 'rectangle':
            return points
        elif shape_type == 'polygon':
            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = 0
            max_y = 0
            for p in points:
                if p[0] < min_x:
                    min_x = p[0]
                if p[0] > max_x:
                    max_x = p[0]
                if p[1] < min_y:
                    min_y = p[1]
                if p[1] > max_y:
                    max_y = p[1]
            return [[min_x, min_y], [max_x, max_y]]
        else:
            raise Exception('Unknown shape type {}'.format(shape_type))

