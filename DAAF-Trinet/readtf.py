import tensorflow as tf
import numpy as np
import os
import scipy.misc
from six.moves import cPickle
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import matplotlib.pyplot as plt
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/home/data/usrs/chenyf/Market_train.tfrecord'])
_, serialized_example = reader.read(filename_queue)
keys_to_features = {
    'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/heatmap':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/filename':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/height':
        tf.FixedLenFeature((), tf.int64, 1),
    'image/width':
        tf.FixedLenFeature((), tf.int64, 1),
    'image/keypointnumber': 
        tf.FixedLenFeature((), tf.int64, 1),
    }
features = tf.parse_single_example(serialized_example, keys_to_features)
image = tf.decode_raw(features['image/encoded'], tf.uint8)
heatmap_channel = tf.cast(features['image/keypointnumber'], tf.int32)
label = tf.decode_raw(features['image/heatmap'], tf.uint8)
height = tf.cast(features['image/height'], tf.int32)
width = tf.cast(features['image/width'], tf.int32)
image = tf.reshape(image, [height, width, 3])
#image = tf.cast(image, tf.float32)
label = tf.reshape(label, [height, width, heatmap_channel])
#label = tf.cast(label, tf.float32)
filename = tf.cast(features['image/filename'], tf.string)
#filename = tf.convert_to_tensor(filename)

sess = tf.Session() 
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess= sess, coord= coord)
img_dir='/home/chenyf/dataset/Market'
for i in range(11320):
    images, labels, fn_img = sess.run([image, label, filename])
    fn_lab = str(fn_img[:-4]) + '.png'
    full_img = os.path.join(img_dir, str(fn_img))
    full_lab = os.path.join(img_dir, fn_lab)
    scipy.misc.imsave(full_img, images)
    scipy.misc.imsave(full_lab, labels)
    '''
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(images)
    plt.subplot(1,2,2)
    #labels = np.array(labels/labels.max()*255,dtype=np.uint8)
    plt.imshow(labels)
    plt.show()
    '''

    
sess.close()
coord.request_stop()
coord.join(threads)