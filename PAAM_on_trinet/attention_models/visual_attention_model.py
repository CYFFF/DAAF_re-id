import tensorflow as tf
from nets import resnet_utils

slim = tf.contrib.slim

from nets.resnet_v1 import resnet_v1_50_block_4, resnet_v1_50_block_3_4, resnet_v1_50_block_2_3_4, resnet_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]


def resnet_block_4(inputs, is_training=True):
    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = resnet_v1_50_block_4(inputs, num_classes=None, is_training=is_training, global_pool=True)

    # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
    #     endpoints['resnet_v1_50_block_4/block4'], [1, 2], name='pool5', keep_dims=False)

    net = endpoints['Resnet_block_4/block4']
    # return endpoints, 'resnet_v1_50_block_4'

    return net


def resnet_block_3_4(inputs, is_training=True):
    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = resnet_v1_50_block_3_4(inputs, num_classes=None, is_training=is_training, global_pool=True)

    # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
    #     endpoints['resnet_v1_50_block_4/block4'], [1, 2], name='pool5', keep_dims=False)

    net = endpoints['Resnet_block_3_4/block4']
    return net


def resnet_block_2_3_4(inputs, is_training=True):
    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = resnet_v1_50_block_2_3_4(inputs, num_classes=None, is_training=is_training, global_pool=True)

    # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
    #     endpoints['resnet_v1_50_block_4/block4'], [1, 2], name='pool5', keep_dims=False)

    net = endpoints['Resnet_block_2_3_4/block4']
    return net


def hmnet_layer_0(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('heatmap_layer_0'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='deconv1')
    return net


def hmnet_layer_1(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('heatmap_layer_1'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='deconv2')
    return net


def hmnet_layer_2(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('heatmap_layer_2'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='deconv3')
    return net


def hmnet_layer_3(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('heatmap_layer_3'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', activation_fn=None,
                                        scope='deconv4')
    return net


def hmnet_layer_4(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('heatmap_layer_4'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1')
    return net


def loss(feature, label, image_size):
    heatmap = tf.image.resize_images(feature, image_size)

    tf.summary.image('heatmap_0', tf.gather(heatmap, [0], axis=3), 1)
    tf.summary.image('label_0', tf.gather(label, [0], axis=3), 1)
    tf.summary.image('heatmap_1', tf.gather(heatmap, [1], axis=3), 1)
    tf.summary.image('label_1', tf.gather(label, [1], axis=3), 1)
    tf.summary.image('heatmap_2', tf.gather(heatmap, [2], axis=3), 1)
    tf.summary.image('label_2', tf.gather(label, [2], axis=3), 1)
    tf.summary.image('heatmap_3', tf.gather(heatmap, [3], axis=3), 1)
    tf.summary.image('label_3', tf.gather(label, [3], axis=3), 1)
    tf.summary.image('heatmap_4', tf.gather(heatmap, [4], axis=3), 1)
    tf.summary.image('label_4', tf.gather(label, [4], axis=3), 1)
    tf.summary.image('heatmap_4', tf.gather(heatmap, [5], axis=3), 1)
    tf.summary.image('label_4', tf.gather(label, [5], axis=3), 1)

    loss = tf.nn.l2_loss(heatmap - label)
    return loss


def loss_mutilayer(heatmap_out_layer_0, heatmap_out_layer_1, heatmap_out_layer_2, heatmap_out_layer_3,
                   heatmap_out_layer_4, labels, net_input_size):
    labels = (tf.sign(labels - 100.5) + 1) / 2

    heatmap_gt_0 = tf.image.resize_images(labels, [16, 8])
    heatmap_gt_1 = tf.image.resize_images(labels, [32, 16])
    heatmap_gt_2 = tf.image.resize_images(labels, [64, 32])
    heatmap_gt_3 = tf.image.resize_images(labels, [128, 64])

    kp_num = 1
    heatmap_pre_0 = slim.conv2d(heatmap_out_layer_0, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid,
                                scope='conv1x1_01')
    heatmap_pre_1 = slim.conv2d(heatmap_out_layer_1, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid,
                                scope='conv1x1_02')
    heatmap_pre_2 = slim.conv2d(heatmap_out_layer_2, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid,
                                scope='conv1x1_03')
    heatmap_pre_3 = heatmap_out_layer_4

    tf.summary.image('heatmap_pre_0', tf.gather(heatmap_pre_0, [0], axis=3), 1)
    tf.summary.image('heatmap_gt_0', tf.gather(heatmap_gt_0, [0], axis=3), 1)
    tf.summary.image('heatmap_pre_1', tf.gather(heatmap_pre_1, [0], axis=3), 1)
    tf.summary.image('heatmap_gt_1', tf.gather(heatmap_gt_1, [0], axis=3), 1)
    tf.summary.image('heatmap_pre_2', tf.gather(heatmap_pre_2, [0], axis=3), 1)
    tf.summary.image('heatmap_gt_2', tf.gather(heatmap_gt_2, [0], axis=3), 1)
    tf.summary.image('heatmap_pre_3', tf.gather(heatmap_pre_3, [0], axis=3), 1)
    tf.summary.image('heatmap_gt_3', tf.gather(heatmap_gt_3, [0], axis=3), 1)

    loss = tf.reduce_mean(tf.nn.l2_loss(heatmap_pre_3 - heatmap_gt_3))
    return loss



