""" A bunch of general utilities shared by train/embed/eval """

from argparse import ArgumentTypeError
import logging
import os

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
import tensorflow as tf_v2
from six.moves import cPickle

# Commandline argument parsing

import scipy.io as sio
###

def check_directory(arg, access=os.W_OK, access_str="writeable"):
    """ Check for directory-type argument validity.

    Checks whether the given `arg` commandline argument is either a readable
    existing directory, or a createable/writeable directory.

    Args:
        arg (string): The commandline argument to check.
        access (constant): What access rights to the directory are requested.
        access_str (string): Used for the error message.

    Returns:
        The string passed din `arg` if the checks succeed.

    Raises:
        ArgumentTypeError if the checks fail.
    """
    path_head = arg
    while path_head:
        if os.path.exists(path_head):
            if os.access(path_head, access):
                # Seems legit, but it still doesn't guarantee a valid path.
                # We'll just go with it for now though.
                return arg
            else:
                raise ArgumentTypeError(
                    'The provided string `{0}` is not a valid {1} path '
                    'since {2} is an existing folder without {1} access.'
                    ''.format(arg, access_str, path_head))
        path_head, _ = os.path.split(path_head)

    # No part of the provided string exists and can be written on.
    raise ArgumentTypeError('The provided string `{}` is not a valid {}'
                            ' path.'.format(arg, access_str))


def writeable_directory(arg):
    """ To be used as a type for `ArgumentParser.add_argument`. """
    return check_directory(arg, os.W_OK, "writeable")


def readable_directory(arg):
    """ To be used as a type for `ArgumentParser.add_argument`. """
    return check_directory(arg, os.R_OK, "readable")


def number_greater_x(arg, type_, x):
    try:
        value = type_(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, type_.__name__))

    if value > x:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than {} was '
            'required'.format(arg, type_.__name__, x))


def positive_int(arg):
    return number_greater_x(arg, int, 0)


def nonnegative_int(arg):
    return number_greater_x(arg, int, -1)


def positive_float(arg):
    return number_greater_x(arg, float, 0)


def float_or_string(arg):
    """Tries to convert the string to float, otherwise returns the string."""
    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg


# Dataset handling
###


def load_dataset(csv_file, image_root, fail_on_missing=True, is_train=False):
    """ Loads a dataset .csv file, returning PIDs and FIDs.

    PIDs are the "person IDs", i.e. class names/labels.
    FIDs are the "file IDs", which are individual relative filenames.

    Args:
        csv_file (string, file-like object): The csv data file to load.
        image_root (string): The path to which the image files as stored in the
            csv file are relative to. Used for verification purposes.
            If this is `None`, no verification at all is made.
        fail_on_missing (bool or None): If one or more files from the dataset
            are not present in the `image_root`, either raise an IOError (if
            True) or remove it from the returned dataset (if False).

    Returns:
        (pids, fids) a tuple of numpy string arrays corresponding to the PIDs,
        i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

    Raises:
        IOError if any one file is missing and `fail_on_missing` is True.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    pids, fids = dataset.T

    # Possibly check if all files exist
    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            if is_train:
                real_fid = fid[:-4]
                # real_fid = fid
            else:
                real_fid = fid[:-4]
            missing[i] = not os.path.isfile(os.path.join(image_root, real_fid+'.jpg'))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                            '{} images are missing'.format(
                                csv_file, image_root, missing_count, len(fids)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                    ' dataset.'.format(missing_count))
                # We simply remove the missing files.
                fids = fids[np.logical_not(missing)]
                pids = pids[np.logical_not(missing)]

    # train_img_path = []
    # train_person_id = []
    # train_camera_id = []
    # file_list = os.listdir(os.path.join(image_root, 'bounding_box_train'))
    # # file_list = sorted(file_list)
    # file_list.sort()
    # for file_name in file_list:
    #     path = 'bounding_box_train/' + file_name
    #     train_img_path.append(path)
    #     train_person_id.append(file_name[0:4])
    #     train_camera_id.append(file_name[6])
    #
    # pids = np.array(train_person_id)
    # fids = np.array(train_img_path)
    # cids = np.array(train_camera_id)
    #
    # pids_num = pids.astype(int)
    # sio.savemat('pids_train.mat', {'pids_train': pids_num})
    # cids_num = cids.astype(int)
    # sio.savemat('cids_train.mat', {'cids_train': cids_num})
    #
    # if image_root is not None:
    #     missing = np.full(len(fids), False, dtype=bool)
    #     for i, fid in enumerate(fids):
    #         missing[i] = not os.path.isfile(os.path.join(image_root, fid))
    #
    #     missing_count = np.sum(missing)
    #     if missing_count > 0:
    #         if fail_on_missing:
    #             raise IOError('Using the `{}` file and `{}` as an image root {}/'
    #                           '{} images are missing'.format(
    #                 csv_file, image_root, missing_count, len(fids)))
    #         else:
    #             print('[Warning] removing {} missing file(s) from the'
    #                   ' dataset.'.format(missing_count))
    #             # We simply remove the missing files.
    #             fids = fids[np.logical_not(missing)]
    #             pids = pids[np.logical_not(missing)]


    return pids, fids

def load_dataset_test(csv_file, image_root, fail_on_missing=False):
    train_img_path = []
    train_person_id = []
    train_camera_id = []
    file_list = os.listdir(os.path.join(image_root, 'bounding_box_test'))
    file_list.sort()
    for file_name in file_list:
        path = 'bounding_box_test/' + file_name
        train_img_path.append(path)
        train_person_id.append(file_name[0:4])
        train_camera_id.append(file_name[6])

    pids = np.array(train_person_id)
    fids = np.array(train_img_path)
    cids = np.array(train_camera_id)


    # pids_num = pids.astype(int)
    # sio.savemat('pids_test.mat', {'pids_test': pids_num})
    # cids_num = cids.astype(int)
    # sio.savemat('cids_test.mat', {'cids_test': cids_num})

    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            missing[i] = not os.path.isfile(os.path.join(image_root, fid))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                              '{} images are missing'.format(
                    csv_file, image_root, missing_count, len(fids)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                      ' dataset.'.format(missing_count))
                # We simply remove the missing files.
                fids = fids[np.logical_not(missing)]
                pids = pids[np.logical_not(missing)]
    return pids, fids


def load_dataset_query(csv_file, image_root, fail_on_missing=False):
    train_img_path = []
    train_person_id = []
    train_camera_id = []
    file_list = os.listdir(os.path.join(image_root, 'query'))
    file_list.sort()
    for file_name in file_list:
        path = 'query/' + file_name
        train_img_path.append(path)
        train_person_id.append(file_name[0:4])
        train_camera_id.append(file_name[6])

    pids = np.array(train_person_id)
    fids = np.array(train_img_path)
    cids = np.array(train_camera_id)

    # pids_num = pids.astype(int)
    # sio.savemat('pids_query.mat', {'pids_query': pids_num})
    # cids_num = cids.astype(int)
    # sio.savemat('cids_query.mat', {'cids_query': cids_num})

    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            missing[i] = not os.path.isfile(os.path.join(image_root, fid))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                              '{} images are missing'.format(
                    csv_file, image_root, missing_count, len(fids)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                      ' dataset.'.format(missing_count))
                # We simply remove the missing files.
                fids = fids[np.logical_not(missing)]
                pids = pids[np.logical_not(missing)]
    return pids, fids


def fid_to_image(fid, pid, image_root, image_size):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))
    
    #image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid, '.jpg']))
    
    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf_v2.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf_v2.image.resize(image_decoded, image_size)

    return image_resized, fid, pid

def fid_to_image_label(fid, pid, image_root, image_size):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.

    image_encoded = tf.io.read_file(tf.reduce_join([image_root, '/', fid]))
    #image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))

    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf_v2.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf_v2.image.resize(image_decoded, image_size)

    # keypt_root = '/data1/chenyf/cuhk03-np-keypoints/labeled/'
    keypt_root = '/data/chenyifan/Market_cpn_keypoints/'
    temp = tf_v2.strings.regex_replace(fid,'.jpg','')
    temp = tf_v2.strings.regex_replace(temp,'bounding_box_train', 'bounding_box_train_256')

    for i in range(17):
        keypt_encoded_temp = tf.io.read_file(tf_v2.strings.reduce_join([keypt_root, temp, '_' + '%02d' % (i) + '.png']))
        keypt_decoded_temp = tf_v2.io.decode_jpeg(keypt_encoded_temp, channels=1)
        keypt_resized_temp = tf.image.resize(keypt_decoded_temp, image_size)

        if i == 0:
            keypt_resized = keypt_resized_temp
        else:
            keypt_resized = tf.concat([keypt_resized, keypt_resized_temp], axis=2)

    mask_encoded = tf.io.read_file(tf_v2.strings.reduce_join(['/data/chenyifan/mask-anno/', fid]))
    mask_decoded = tf_v2.io.decode_jpeg(mask_encoded, channels=1)
    mask_resized = tf.image.resize(mask_decoded, image_size)

    # print(1234)
    return image_resized, keypt_resized, mask_resized, fid, pid


def fid_to_image_attribute(fid, pid, attr_lookuptable, image_root, image_size):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    # if not hasattr(fid_to_image_attribute, "_label"):


    image_encoded = tf.io.read_file(tf.reduce_join([image_root, '/', fid]))
    #image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))

    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf_v2.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf_v2.image.resize(image_decoded, image_size)

    id = tf_v2.strings.substr(fid, 19, 4)

    lookuptable = attr_lookuptable[0]
    relative_id = lookuptable.lookup(id)

    attribute_label = []
    for i in range(1,28):
        attribute_label.append(attr_lookuptable[i].lookup(relative_id))

    return image_resized, attribute_label, fid, pid


def split(im, fid, pid):
    split0, split1, split2 = tf.split(im, [3, 17, 1], 2)
    # print(12333)
    return split0, split1, split2, fid, pid

def get_logging_dict(name):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'common.ColorStreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': name + '.log',
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },

            # extra ones to shut up.
            'tensorflow': {
                'handlers': ['stderr', 'logfile'],
                'level': 'INFO',
            },
        }
    }


# Source for the remainder: https://gist.github.com/mooware/a1ed40987b6cc9ab9c65
# Fixed some things mentioned in the comments there.

# colored stream handler for python logging framework (use the ColorStreamHandler class).
#
# based on:
# http://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/1336640#1336640

# how to use:
# i used a dict-based logging configuration, not sure what else would work.
#
# import logging, logging.config, colorstreamhandler
#
# _LOGCONFIG = {
#     "version": 1,
#     "disable_existing_loggers": False,
#
#     "handlers": {
#         "console": {
#             "class": "colorstreamhandler.ColorStreamHandler",
#             "stream": "ext://sys.stderr",
#             "level": "INFO"
#         }
#     },
#
#     "root": {
#         "level": "INFO",
#         "handlers": ["console"]
#     }
# }
#
# logging.config.dictConfig(_LOGCONFIG)
# mylogger = logging.getLogger("mylogger")
# mylogger.warning("foobar")

# Copyright (c) 2014 Markus Pointner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

class _AnsiColorStreamHandler(logging.StreamHandler):
    DEFAULT = '\x1b[0m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    CYAN    = '\x1b[36m'

    CRITICAL = RED
    ERROR    = RED
    WARNING  = YELLOW
    INFO     = DEFAULT  # GREEN
    DEBUG    = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return (color + text + self.DEFAULT) if self.is_tty() else text

    def is_tty(self):
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty()


class _WinColorStreamHandler(logging.StreamHandler):
    # wincon.h
    FOREGROUND_BLACK     = 0x0000
    FOREGROUND_BLUE      = 0x0001
    FOREGROUND_GREEN     = 0x0002
    FOREGROUND_CYAN      = 0x0003
    FOREGROUND_RED       = 0x0004
    FOREGROUND_MAGENTA   = 0x0005
    FOREGROUND_YELLOW    = 0x0006
    FOREGROUND_GREY      = 0x0007
    FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified.
    FOREGROUND_WHITE     = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

    BACKGROUND_BLACK     = 0x0000
    BACKGROUND_BLUE      = 0x0010
    BACKGROUND_GREEN     = 0x0020
    BACKGROUND_CYAN      = 0x0030
    BACKGROUND_RED       = 0x0040
    BACKGROUND_MAGENTA   = 0x0050
    BACKGROUND_YELLOW    = 0x0060
    BACKGROUND_GREY      = 0x0070
    BACKGROUND_INTENSITY = 0x0080 # background color is intensified.

    DEFAULT  = FOREGROUND_WHITE
    CRITICAL = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
    ERROR    = FOREGROUND_RED | FOREGROUND_INTENSITY
    WARNING  = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
    INFO     = FOREGROUND_GREEN
    DEBUG    = FOREGROUND_CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def _set_color(self, code):
        import ctypes
        ctypes.windll.kernel32.SetConsoleTextAttribute(self._outhdl, code)

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)
        # get file handle for the stream
        import ctypes, ctypes.util
        # for some reason find_msvcrt() sometimes doesn't find msvcrt.dll on my system?
        crtname = ctypes.util.find_msvcrt()
        if not crtname:
            crtname = ctypes.util.find_library("msvcrt")
        crtlib = ctypes.cdll.LoadLibrary(crtname)
        self._outhdl = crtlib._get_osfhandle(self.stream.fileno())

    def emit(self, record):
        color = self._get_color(record.levelno)
        self._set_color(color)
        logging.StreamHandler.emit(self, record)
        self._set_color(self.FOREGROUND_WHITE)

# select ColorStreamHandler based on platform
import platform
if platform.system() == 'Windows':
    ColorStreamHandler = _WinColorStreamHandler
else:
    ColorStreamHandler = _AnsiColorStreamHandler


    
