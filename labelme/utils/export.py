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

from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme.utils.map import Map
from labelme.windows import Export

# https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html#recordfiledetection-for-entire-dataset-packed-in-single-mxnet-recordfile
def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(int(labels[0]))] + [str(x) for x in labels[1:]]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

def make_lst_file(export_file, label_files, label_list_file, progress, validation_ratio=0.0):
    # Update progress bar
    progress.setLabelText(_('Creating lst file ...'))

    export_folder = os.path.dirname(export_file)
    export_file_name = os.path.splitext(os.path.basename(export_file))[0]
    num_label_files = len(label_files)
    lst_file_train = os.path.normpath(os.path.join(export_folder, '{}_train.lst'.format(export_file_name)))
    lst_file_val = os.path.normpath(os.path.join(export_folder, '{}_val.lst'.format(export_file_name)))

    # Get all labels with index
    label_list = None
    with open(label_list_file) as f:
        label_list = f.read().split('\n')
    if label_list is None:
        logger.error('No labels found in label list file: {}'.format(label_list_files))
        return
    label_to_idx = get_label_to_idx_dict(label_list)

    # Shuffle
    random.seed(42)
    random.shuffle(label_files)

    # Create lst files for training/validation
    size_train = int(num_label_files * (1.0 - validation_ratio))
    label_files_train = label_files[:size_train]
    label_files_val = label_files[size_train:]
    write_lst_file(lst_file_train, label_files_train, label_to_idx, progress)
    write_lst_file(lst_file_val, label_files_val, label_to_idx, progress)

    return (lst_file_train, len(label_files_train)), (lst_file_val, len(label_files_val))

def write_lst_file(lst_file, label_files, label_to_idx, progress):
    num_label_files = len(label_files)
    start_value = progress.value()
    # Open (new) lst file
    with open(lst_file, 'w+') as f:
        for idx in range(num_label_files):
            if progress.wasCanceled():
                return
            label_file = LabelFile(label_files[idx])
            shapes = [{'label': s[0], 'points': s[1], 'type': s[4]} for s in label_file.shapes]
            # Skip files without shapes
            if len(shapes) == 0:
                continue
            channels = 3 # Assume 3 channels
            im_shape = (label_file.imageHeight, label_file.imageWidth, channels)
            boxes = []
            labels = []
            for shape in shapes:
                box = []
                points = shape['points']
                if shape['type'] is not 'rectangle':
                    # Convert polygons to rectangle
                    points = shape_points_to_rectangle(shape['type'], shape['points'])
                for point in points:
                    box = box + point
                boxes.append(box)
                labels.append(shape['label'])
            ids = np.array([label_to_idx[l] for l in labels], dtype=np.int32)
            boxes = np.array(boxes)
            line = write_line(label_file.imagePath, im_shape, boxes, ids, idx)
            f.write(line)
            # Update progress bar
            progress.setValue(idx + start_value + 1)

def make_label_list(label_list_file, label_files):
    num_label_files = len(label_files)
    label_list = []
    for idx in range(num_label_files):
        label_file = LabelFile(label_files[idx])
        shapes = [s[0] for s in label_file.shapes]
        for s in shapes:
            label_list.append(s)
    label_list = list(set(label_list))
    label_list.sort()
    logger.debug('Found {} labels in dataset: {}'.format(len(label_list), label_list))
    with open(label_list_file, 'w+') as f:
        f.write('\n'.join(label_list))

def lst2rec(prefix, root, progress, num_label_files, pass_through=False, resize=0, 
    center_crop=False, quality=95, encoding='.jpg', pack_label=False):

    # Restore translation function '_' as it gets overwritten somehow
    _ = builtins._

    args = Map({
        'prefix': prefix,
        'root': root,
        'pass_through': pass_through,
        'resize': resize,
        'center_crop': center_crop,
        'quality': quality,
        'encoding': encoding,
        'pack_label': pack_label,
    })

    if os.path.isdir(args.prefix):
        working_dir = args.prefix
    else:
        working_dir = os.path.dirname(args.prefix)
    files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                if os.path.isfile(os.path.join(working_dir, fname))]
    count = 0
    for fname in files:
        if fname.startswith(args.prefix) and fname.endswith('.lst'):
            print('Creating', Export.config('extensions')['imagerecord'], 'file from', fname, 'in', working_dir)
            count += 1
            image_list = read_list(fname)

            # Update progress bar
            start_value = progress.value()
            progress.setLabelText(_('Creating rec file ...'))

            # -- write_record -- #
            try:
                import Queue as queue
            except ImportError:
                import queue
            q_out = queue.Queue()
            fname = os.path.basename(fname)
            fname_rec = os.path.splitext(fname)[0] + Export.config('extensions')['imagerecord']
            fname_idx = os.path.splitext(fname)[0] + '.idx'
            record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx), os.path.join(working_dir, fname_rec), 'w')
            cnt = 0
            pre_time = time.time()
            for i, item in enumerate(image_list):
                image_encode(args, i, item, q_out)
                if q_out.empty():
                    continue
                _, s, _ = q_out.get()
                record.write_idx(item[0], s)
                if cnt % 1000 == 0:
                    cur_time = time.time()
                    logger.debug('time: {} count: {}'.format(cur_time - pre_time, cnt))
                    pre_time = cur_time
                cnt += 1
                progress.setValue(i + start_value + 1)
                if progress.wasCanceled():
                    return None
            return os.path.join(working_dir, fname_rec)
    if not count:
        logger.debug('Did not find and list file with prefix {}'.format(args.prefix))

    return None

def read_list(path_in):
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
                print('lst should have at least has three parts, but only has {} parts for {}}'.format(line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for {}, detail: {}'.format(line, e))
                continue
            yield item

def image_encode(args, i, item, q_out):
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
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = Image.open(fullpath)
    except:
        traceback.print_exc()
        print('imread error trying to load file: {}'.format(fullpath))
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: {}'.format(fullpath))
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
        print('pack_img error on file: {} {}'.format(fullpath, e))
        q_out.put((i, None, item))
        return

def get_label_to_idx_dict(label_list):
    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx
    return label_dict

def shape_points_to_rectangle(shape_type, points):
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