# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import skimage
import logging
import os

from lxml import etree
import PIL.Image
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import dataset_util
#import label_map_util

#os.environ['CUDA_VISIBLE_DEVICES']='0,2'
flags = tf.app.flags
flags.DEFINE_string('xml_dir', '', 'Root directory to xml.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('datasetFlag', '', 'required a dataset flag')
flags.DEFINE_integer('kpNum', 0, 'required a kp num')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']

_all_num = 0

def dict_to_tf_example(writer, data):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = os.path.join(data['folder'], data['filename']).replace('/home/data/usrs/jiangyz/images', '/data1/chenyf')
  #print('full_path%s'%full_path)

  OriImg = PIL.Image.open(full_path)
  if (OriImg.mode != 'RGB'):
    OriImg = OriImg.convert("RGB")
    print(full_path + ' is not a rgb image, converting...')
  OriImgArray = np.asarray(OriImg)
    
  w = int(OriImgArray.shape[1])
  h = int(OriImgArray.shape[0])

  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if difficult:
      print('there is a difficult instance.....')
      raw_input()
      continue

    left = int(obj['bndbox']['xmin'])
    top = int(obj['bndbox']['ymin'])
    right = int(obj['bndbox']['xmax'])
    down = int(obj['bndbox']['ymax'])
    # if (right-left)*(down-top)<w*h/4:
    #   continue

    difficult_obj = int(difficult)
    imgSinglePerson = OriImg
    imgSingle = np.asarray(imgSinglePerson)
 
    img_raw = imgSingle.tostring()
    classes_text = obj['name'].encode('utf8')
    classes = 0
    
    kp_cor_v = [int(x) for x in obj['keypoints']['visible']]
    
    truncated = int(obj['truncated'])
    poses = obj['pose'].encode('utf8')

    
    kpNum=FLAGS.kpNum
     
    if(sum(1 for x in kp_cor_v if x) < (kpNum+1)/2):
      continue

    kp_cor = []
    for tmp_id in range(kpNum):
      if kp_cor_v[tmp_id] != 0:
        #convert to imgSinglePerson
        xc = int(obj['keypoints']['x'][tmp_id])
        yc = int(obj['keypoints']['y'][tmp_id])
        kp_cor.append((xc, yc))
    global _all_num
    _all_num += 1
    #tf.summary.image('image',tf.convert_to_tensor(np.array([OriImgArray])),1)
    #show images
    
    #color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(0,0,0)]
    heatmap = np.zeros([h, w, kpNum], np.float32)
    #print('shape of heatmap=',np.shape(heatmap))
    sigma = 10
    for idx, cor_xy in enumerate(kp_cor):
      cor_x, cor_y = cor_xy
      r = 36                          # int(8/96.0*224)
      for ii in range(-r, r+1, 1):
        for jj in range(-r, r+1, 1):
          xxxx = cor_x + ii
          yyyy = cor_y + jj
          if(xxxx < 0)or(yyyy < 0)or(xxxx > w-1)or(yyyy>h-1):
            continue
          heatmap[yyyy, xxxx, idx] += np.exp(-(ii*ii+jj*jj)/(2*sigma*sigma))
          #heatmap[yyyy,xxxx]=255
    heatmap[heatmap>1] = 1.0

    #print('length of heatmap=',len(hm_raw))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(h),
        'image/width': dataset_util.int64_feature(w),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(img_raw),
        'image/heatmap': dataset_util.float_list_feature(heatmap.flatten()),
        'image/keypointnumber': dataset_util.int64_feature(kpNum),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/class/text': dataset_util.bytes_feature(classes_text),
        'image/object/class/label': dataset_util.int64_feature(classes),
        'image/object/difficult': dataset_util.int64_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_feature(truncated),
        'image/object/view': dataset_util.bytes_feature(poses),
    }))
    writer.write(example.SerializeToString())


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  #print('hj')
  #raw_input()
  xmlDir = FLAGS.xml_dir
  logging.info('Reading from %s .', xmlDir)

  if FLAGS.datasetFlag == 'custom':
    xml_path_ = get_all_path_iterative(xmlDir)
    print('process custom data...')
#    raw_input()
  elif FLAGS.datasetFlag == 'coco':
    xml_path_ = []
    xmlList = os.listdir(xmlDir)
    for xmlInstance in xmlList:
      print(xmlInstance)
      xmlPath = os.path.join(xmlDir, xmlInstance)
      xml_path_.append(xmlPath)
  else:
    print('dataset flag error...')
#    raw_input()

  print('find xml num: %d'%len(xml_path_))
#  raw_input() 
  xmlId = 0
  for xmlPath in xml_path_:
    #print(xmlId)
    if xmlId % 1000 == 0:
      logging.info('On xml %d of %d', xmlId, len(xml_path_))
      print('On xml %d of %d', xmlId, len(xml_path_))
    #print('xmlPath %s'%xmlPath)
    with tf.gfile.GFile(xmlPath, 'r') as fid:
      xmlStr = fid.read()
    xmlTree = etree.fromstring(xmlStr)
    xmlData = dataset_util.recursive_parse_xml_to_dict(xmlTree)['annotation']
    
    #if not xmlData['object']:
    if 'object' not in xmlData:
      print('skip, there is no object in the xml...... %s'%xmlPath)
      #raw_input()
      continue
    dict_to_tf_example(writer, xmlData)
    #if(_all_num > 30100):
    #  break
    xmlId += 1

  print("all xml num: %d"%xmlId)
  print('all good person num: %d'%_all_num)
  writer.close()
  
def get_all_path_iterative(rootdir):
  list = os.listdir(rootdir)
  xml_path = []
  for line in list:
    filepath = os.path.join(rootdir, line)
    # if os.path.isdir(filepath):
    #   print("sub dir: " + filepath)
    #   xmlList = os.listdir(filepath)
    #   for xml in xmlList:
    #     xmlpath = os.path.join(filepath,xml)
    #     xml_path.append(xmlpath)
    if ~os.path.isdir(filepath):
      xml_path.append(filepath)
  return xml_path


if __name__ == '__main__':
  tf.app.run()
