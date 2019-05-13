import numpy as np
import os
import sys
import itertools

from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme.scripts import im2rec
from labelme.utils.map import Map
from labelme.utils.im2rec import *

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

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
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

def make_lst_file(export_folder, label_files, progress):
    num_label_files = len(label_files)
    lst_file = os.path.join(export_folder, 'dataset.lst')

    # Update progress bar
    start_value = progress.value()
    progress.setLabelText(_('Creating lst file ...'))

    # Get all labels with index
    label_to_idx = get_label_to_idx_dict(label_files)

    # Open (new) lst file
    with open(lst_file, 'w+') as f:
        for idx in range(num_label_files):
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
            ids = np.array([label_to_idx[l] for l in labels])
            boxes = np.array(boxes)
            line = write_line(label_file.imagePath, im_shape, boxes, ids, idx)
            f.write(line)
            # Update progress bar
            progress.setValue(idx + start_value + 1)

    return lst_file

def im2rec(prefix, root, progress, num_label_files, list=False, exts=['.jpeg', '.jpg', '.png'], chunks=1, train_ratio=1.0, 
    test_ratio=0.0, recursive=False, no_shuffle=True, pass_through=False, resize=0, 
    center_crop=False, quality=95, num_thread=1, color=1, encoding='.jpg', pack_label=False):
    args = Map({
        'prefix': prefix,
        'root': root,
        'list': list,
        'exts': exts,
        'chunks': chunks,
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'recursive': recursive,
        'no_shuffle': no_shuffle,
        'pass_through': pass_through,
        'resize': resize,
        'center_crop': center_crop,
        'quality': quality,
        'num_thread': num_thread,
        'color': color,
        'encoding': encoding,
        'pack_label': pack_label,
    })

    # if the '--list' is used, it generates .lst file
    if args.list:
        make_list(args)
    # otherwise read .lst file to generates .rec file
    else:
        if os.path.isdir(args.prefix):
            working_dir = args.prefix
        else:
            working_dir = os.path.dirname(args.prefix)
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                    if os.path.isfile(os.path.join(working_dir, fname))]
        count = 0
        for fname in files:
            if fname.startswith(args.prefix) and fname.endswith('.lst'):
                print('Creating .rec file from', fname, 'in', working_dir)
                count += 1
                image_list = read_list(fname)

                # Update progress bar
                start_value = progress.value()
                progress.setLabelText('Creating rec file ...') # no translation possible with _()

                # -- write_record -- #
                if args.num_thread > 1 and multiprocessing is not None:
                    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                    q_out = multiprocessing.Queue(1024)
                    # define the process
                    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                    for i in range(args.num_thread)]
                    # process images with num_thread process
                    for p in read_process:
                        p.start()
                    # only use one process to write .rec to avoid race-condtion
                    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                    write_process.start()
                    # put the image list into input queue
                    for i, item in enumerate(image_list):
                        q_in[i % len(q_in)].put((i, item))
                    for q in q_in:
                        q.put(None)
                    for p in read_process:
                        p.join()

                    q_out.put(None)
                    write_process.join()
                else:
                    print('multiprocessing not available, fall back to single threaded encoding')
                    try:
                        import Queue as queue
                    except ImportError:
                        import queue
                    q_out = queue.Queue()
                    fname = os.path.basename(fname)
                    fname_rec = os.path.splitext(fname)[0] + '.rec'
                    fname_idx = os.path.splitext(fname)[0] + '.idx'
                    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                           os.path.join(working_dir, fname_rec), 'w')
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
                            print('time:', cur_time - pre_time, ' count:', cnt)
                            pre_time = cur_time
                        cnt += 1
                        progress.setValue(i + start_value + 1)
        if not count:
            print('Did not find and list file with prefix %s'%args.prefix)

def get_label_to_idx_dict(label_files):
    num_label_files = len(label_files)
    label_set = set()
    for idx in range(num_label_files):
        label_file = LabelFile(label_files[idx])
        shapes = [s[0] for s in label_file.shapes]
        for s in shapes:
            label_set.add(s)
    label_list = list(label_set)
    label_list.sort()
    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx
    logger.debug('Found {} labels in dataset: {}'.format(len(label_list), label_dict))
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