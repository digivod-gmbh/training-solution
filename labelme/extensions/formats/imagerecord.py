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
import gluoncv as gcv

from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme.utils.map import Map
from labelme.config.export import Export

from .format import DatasetFormat
from .intermediate import IntermediateFormat


class FormatImageRecord(DatasetFormat):

    _files = {}
    _format = 'imagerecord'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.num_samples = -1
        FormatImageRecord._files['labels'] = Export.config('labels_file')

    def getOutputFileName(self, split='train'):
        if split == 'train':
            return 'train.rec'
        elif split == 'val':
            return 'val.rec'
        raise Exception('Unknown split {}'.format(split))

    def isValidFormat(self, dataset_folder_or_file):
        if not os.path.isfile(dataset_folder_or_file):
            logger.warning('Dataset file {} does not exist'.format(dataset_folder_or_file))
            return False
        file_dir = os.path.dirname(dataset_folder_or_file)
        file_name = os.path.basename(dataset_folder_or_file)
        base = os.path.splitext(file_name)[0]
        idx_file = os.path.join(file_dir, base + '.idx')
        if not os.path.isfile(idx_file):
            logger.warning('Idx file {} does not exist'.format(idx_file))
            return False
        return True

    def getLabels(self):
        labels = []
        input_folder = os.path.dirname(self.input_folder_or_file)
        label_file = os.path.join(input_folder, FormatImageRecord._files['labels'])
        if os.path.isfile(label_file):
            logger.debug('Load labels from file {}'.format(label_file))
            for i, line in enumerate(open(label_file).readlines()):
                labels.append(line)
        else:
            labels = set([])
            logger.debug('No label file found. Start reading dataset')
            record = mx.recordio.MXRecordIO(self.input_folder_or_file, 'r')
            record.reset()
            self.num_samples = 0
            while True:
                try:
                    item = record.read()
                    if not item:
                        break
                    header, s = mx.recordio.unpack(item)
                    for i in range(4, len(header.label), 5):
                        label_idx = str(header.label[i])
                        labels.append(label_idx)
                        self.num_samples = self.num_samples + 1
                except Exception as e:
                    logger.error(traceback.format_exc())
            record.close()

        return list(labels)

    def getNumSamples(self):
        if self.num_samples == -1:
            logger.debug('Count samples in dataset')
            samples = gcv.data.RecordFileDetection(self.input_folder_or_file)
            self.num_samples = len(samples)
        return self.num_samples

    def getDatasetForTraining(self):
        samples = gcv.data.RecordFileDetection(self.input_folder_or_file)
        self.num_samples = len(samples)
        return samples

    def importFolder(self):
        if self.input_folder_or_file is None:
            raise Exception('Input folder must be initialized for import')
        
        output_folder = self.output_folder

        self.intermediate = IntermediateFormat()
        self.importToIntermediate(self.input_folder_or_file, output_folder)
        self.intermediate.toLabelFiles()

    def importToIntermediate(self, rec_file, output_folder):
        # Labels
        all_labels = []
        input_folder = os.path.dirname(self.input_folder_or_file)
        label_file = os.path.join(input_folder, FormatImageRecord._files['labels'])
        if os.path.isfile(label_file):
            logger.debug('Load labels from file {}'.format(label_file))
            for i, line in enumerate(open(label_file).readlines()):
                all_labels.append(line)
        else:
            logger.warning('No label file found at {}'.format(label_file))

        self.thread.update.emit(_('Loading image record file ...'), 10, -1)
        self.checkAborted()

        file_pos = 0
        file_size = os.path.getsize(rec_file)
        logger.debug('Start loading of image record file {} with size of {} bytes'.format(rec_file, file_size))

        record = mx.recordio.MXRecordIO(rec_file, 'r')
        record.reset()
        while True:
            try:
                self.checkAborted()
                item = record.read()
                if not item:
                    break
                
                file_pos += len(item)
                percentage = file_pos / file_size * 90
                self.thread.update.emit(_('Loading image record file ...'), 10 + percentage, -1)
                self.checkAborted()

                header, image = mx.recordio.unpack_img(item)
                img_file = os.path.join(output_folder, '{:09d}.jpg'.format(header.id))
                cv2.imwrite(img_file, image)
                image_height = image.shape[0]
                image_width = image.shape[1]
                shapes = []
                for i in range(4, len(header.label), 5):
                    label_idx = int(header.label[i])
                    bbox = header.label[i+1:i+5] 
                    label_name = str(label_idx)
                    if label_idx < len(all_labels):
                        label_name = all_labels[label_idx].strip()
                    points = [
                        [int(bbox[0] * image_width), int(bbox[1] * image_height)],
                        [int(bbox[2] * image_width), int(bbox[3] * image_height)],
                    ]
                    # imagerecord has only rectangle shapes
                    self.intermediate.addSample(img_file, (image_height, image_width), label_name, points, 'rectangle')
                    self.checkAborted()

            except Exception as e:
                logger.error(traceback.format_exc())
                raise Exception(e)

        record.close()

    def export(self):
        if self.intermediate is None:
            raise Exception('Intermediate format must be initialized for export')
        
        self.thread.update.emit(_('Gathering samples ...'), -1, -1)
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

        num_max_progress = num_train_samples + num_val_samples + 10

        # train
        self.thread.update.emit(_('Creating training dataset ...'), -1, num_max_progress)
        self.checkAborted()
        train_output_folder = os.path.join(output_folder, 'train')
        if not os.path.isdir(train_output_folder):
            os.makedirs(train_output_folder)
        self.makeLstFile(train_output_folder, 'train.lst', train_samples)
        train_rec_file = self.getOutputFileName('train')
        self.makeRecFile(data_folder, train_output_folder, train_rec_file, 'train.idx', 'train.lst')
        train_label_file = os.path.join(train_output_folder, FormatImageRecord._files['labels'])
        with open(train_label_file, 'w+') as f:
            f.write('\n'.join(labels))

        # validate
        validation_ratio = self.intermediate.getValidationRatio()
        if validation_ratio > 0.0:
            self.thread.update.emit(_('Creating validation dataset ...'), -1, -1)
            self.checkAborted()
            val_output_folder = os.path.join(output_folder, 'val')
            if not os.path.isdir(val_output_folder):
                os.makedirs(val_output_folder)
            self.makeLstFile(val_output_folder, 'val.lst', val_samples)
            val_rec_file = self.getOutputFileName('val')
            self.makeRecFile(data_folder, val_output_folder, val_rec_file, 'val.idx', 'val.lst')
            val_label_file = os.path.join(val_output_folder, FormatImageRecord._files['labels'])
            with open(val_label_file, 'w+') as f:
                f.write('\n'.join(labels))

    def makeLstFile(self, output_folder, file_name, samples):
        lst_file = os.path.join(output_folder, file_name)

        self.checkAborted()

        # group samples by image
        samples_per_image = self.intermediate.getSamplesPerImage(samples)

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
                self.checkAborted()

    def makeRecFile(self, data_folder, output_folder, file_name_rec, file_name_idx, file_name_lst):
        image_list = self.readLstFile(os.path.join(output_folder, file_name_lst))
        record = mx.recordio.MXIndexedRecordIO(os.path.join(output_folder, file_name_idx), os.path.join(output_folder, file_name_rec), 'w')

        self.checkAborted()

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
            self.thread.update.emit(_('Writing dataset ...'), -1, -1)
            self.checkAborted()
        logger.debug('total time: {} total count: {}'.format(time.time() - pre_time, cnt))

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
                self.checkAborted()
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
