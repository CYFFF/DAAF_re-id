import os
import argparse
import tensorflow as tf
import numpy as np
import scipy.misc
from itertools import count
from importlib import import_module
import json
import common
import glob
from nets import NET_CHOICES
from heads import HEAD_CHOICES
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def parse_args():
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--input', help='images file list ', type=str, default='')
    parser.add_argument('--output_root', help='images file list ', type=str, default='')
    parser.add_argument('--net_input_height', help='height', type=int, default=256)
    parser.add_argument('--net_input_width', help='width', type=int, default=128)
    parser.add_argument('--pre_crop_height', help='height', type=int, default=288)
    parser.add_argument('--pre_crop_width', help='width', type=int, default=144)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    parser.add_argument('--loading_threads', help='loading_threads', type=int, default=2)
    parser.add_argument('--experiment_root', help='experiment_root', type=str, default='')
    parser.add_argument('--model_name', help='model name', type=str, default='resnet_v1_50', choices=NET_CHOICES)
    parser.add_argument('--head_name', help='model name', type=str, default='fc1024', choices=HEAD_CHOICES)
    parser.add_argument('--checkpoint', help='path to the checkpoint file', type=str, default='')
    parser.add_argument('--embedding_dim', help='embedding_dim', type=int, default=128)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    image = tf.placeholder(tf.float32, shape=[None, args.net_input_height, args.net_input_width, 3])
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)
    endpoints, body_prefix = model.endpoints(image, is_training=False)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=False)
    if args.checkpoint is None:
        checkpoint = tf.train.latest_checkpoint(args.experiment_root)
    else:
        checkpoint = os.path.join(args.experiment_root, args.checkpoint)
        print('Restoring from checkpoint: {}'.format(checkpoint))

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, checkpoint)
        json_files = [x for x in glob.glob(args.input + '/*.json', recursive=False)]
        for json_file in json_files:
            #if not os.path.exists('/data1/poseTrack2018/posetrack_data/annotations/val/' + os.path.basename(json_file)):
            #    continue
            with open(json_file, 'r') as f:
                data = json.load(f)
            id_file = {}
            for image_obj in data['images']:
                id_file[image_obj['id']] = image_obj['file_name']
            for annotation in data['annotations']:
                image_file_name = '/data2/dataset/poseTrack2018/posetrack_data/' + id_file[annotation['image_id']]
                img = scipy.misc.imread(image_file_name).astype(np.float32)
                bbox = annotation['bbox']
                patch=img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                resized_patch = scipy.misc.imresize(patch,(args.net_input_height, args.net_input_width, 3))
                reshaped_patch = resized_patch.reshape(1, args.net_input_height, args.net_input_width, 3)
                emb = sess.run(endpoints['emb'], feed_dict={image: reshaped_patch})
                annotation['embedding'] = emb[0].tolist()
            with open(args.output_root + os.path.basename(json_file), 'w') as ouput:
                json.dump(data, ouput, indent=4)

if __name__ == '__main__':
    main()
