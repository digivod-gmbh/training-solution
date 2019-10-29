import os.path as osp
import os
import re

import numpy as np
import PIL.Image

from labelme.utils.draw import label_colormap
from labelme.logger import logger


def lblsave(filename, lbl):
    if osp.splitext(filename)[1] != '.png':
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = label_colormap(255)
        lbl_pil.putpalette((colormap * 255).astype(np.uint8).flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )

def deltree(target):
    for d in os.listdir(target):
        try:
            deltree(target + '/' + d)
        except OSError:
            os.remove(target + '/' + d)
    os.rmdir(target)
    logger.debug('Deleted folder {}'.format(target))

def replace_special_chars(filename):
    filename = filename.replace('ä', 'ae')
    filename = filename.replace('ö', 'oe')
    filename = filename.replace('ü', 'ue')
    filename = filename.replace('Ä', 'Ae')
    filename = filename.replace('Ö', 'Oe')
    filename = filename.replace('Ü', 'Ue')
    filename = filename.replace('ß', 'ss')
    return re.sub(r'[^a-zA-Z0-9 \._-]+', '', filename)

def contains_special_chars(filename, is_path=False):
    if is_path:
        subpath = os.path.splitdrive(filename)[1]
        r = re.search(r'[^a-zA-Z0-9\./\\ _-]+', subpath)
    else:
        r = re.search(r'[^a-zA-Z0-9\. _-]+', filename)
    return r is not None

