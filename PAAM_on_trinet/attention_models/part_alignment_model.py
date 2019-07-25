import tensorflow as tf
from nets import resnet_utils
slim = tf.contrib.slim

# from nets.resnet_v1 import resnet_v1_50_block_4, resnet_v1_50_block_3_4, resnet_v1_50_block_2_3_4, resnet_arg_scope
from nets.resnet_v1 import resnet_arg_scope
_RGB_MEAN = [123.68, 116.78, 103.94]


# def resnet_block_4(inputs, is_training=True):
#     with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
#         _, endpoints = resnet_v1_50_block_4(inputs, num_classes=None, is_training=is_training, global_pool=True)
#
#     # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
#     #     endpoints['resnet_v1_50_block_4/block4'], [1, 2], name='pool5', keep_dims=False)
#
#     net = endpoints['Resnet_block_4/block4']
#     # return endpoints, 'resnet_v1_50_block_4'
#
#     return net
#
#
# def resnet_block_3_4(inputs, is_training=True):
#     with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
#         _, endpoints = resnet_v1_50_block_3_4(inputs, num_classes=None, is_training=is_training, global_pool=True)
#
#     # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
#     #     endpoints['resnet_v1_50_block_4/block4'], [1, 2], name='pool5', keep_dims=False)
#
#     net = endpoints['Resnet_block_3_4/block4']
#     return net
#
#
# def resnet_block_2_3_4(inputs, is_training=True):
#     with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
#         _, endpoints = resnet_v1_50_block_2_3_4(inputs, num_classes=None, is_training=is_training, global_pool=True)
#
#     # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
#     #     endpoints['resnet_v1_50_block_4/block4'], [1, 2], name='pool5', keep_dims=False)
#
#     net = endpoints['Resnet_block_2_3_4/block4']
#     return net


def trans_conv_0(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_0', reuse=tf.AUTO_REUSE):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='trans_conv_0')
    return net


def trans_conv_1(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_1', reuse=tf.AUTO_REUSE):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='trans_conv_1')
    return net


def trans_conv_2(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_2', reuse=tf.AUTO_REUSE):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='trans_conv_2')
    return net


def trans_conv_3(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_3', reuse=tf.AUTO_REUSE):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', activation_fn=None,
                                        scope='trans_conv_3')
    return net


def conv_0(inputs, kp_num=5):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_0'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_0')
    return net

def conv_1(inputs, kp_num=5):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_1'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_1')
    return net

def conv_2(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_2'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_2')
    return net

def conv_3(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_3'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_3')
    return net

def conv_4(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_4'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_4')
    return net

def conv_5(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_5'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_5')
    return net

def tran_conv_0(input, kp_num=5):
    out_0 = trans_conv_0(input, kp_num)
    out_1 = trans_conv_1(out_0, kp_num)
    out_2 = trans_conv_2(out_1, kp_num)
    out_3 = trans_conv_3(out_2, kp_num)
    out_4 = conv_0(out_3, kp_num)
    return out_4

def tran_conv_1(input, kp_num=4):
    out_0 = trans_conv_0(input, kp_num)
    out_1 = trans_conv_1(out_0, kp_num)
    out_2 = trans_conv_2(out_1, kp_num)
    out_3 = trans_conv_3(out_2, kp_num)
    out_4 = conv_1(out_3, kp_num)
    return out_4

def tran_conv_2(input, kp_num=4):
    out_0 = trans_conv_0(input, kp_num)
    out_1 = trans_conv_1(out_0, kp_num)
    out_2 = trans_conv_2(out_1, kp_num)
    out_3 = trans_conv_3(out_2, kp_num)
    out_4 = conv_2(out_3, kp_num)
    return out_4

def tran_conv_3(input, kp_num=4):
    out_0 = trans_conv_0(input, kp_num)
    out_1 = trans_conv_1(out_0, kp_num)
    out_2 = trans_conv_2(out_1, kp_num)
    out_3 = trans_conv_3(out_2, kp_num)
    out_4 = conv_3(out_3, kp_num)
    return out_4

def tran_conv_4(input, kp_num=4):
    out_0 = trans_conv_0(input, kp_num)
    out_1 = trans_conv_1(out_0, kp_num)
    out_2 = trans_conv_2(out_1, kp_num)
    out_3 = trans_conv_3(out_2, kp_num)
    out_4 = conv_4(out_3, kp_num)
    return out_4

def tran_conv_5(input, kp_num=4):
    out_0 = trans_conv_0(input, kp_num)
    out_1 = trans_conv_1(out_0, kp_num)
    out_2 = trans_conv_2(out_1, kp_num)
    out_3 = trans_conv_3(out_2, kp_num)
    out_4 = conv_5(out_3, kp_num)
    return out_4

def keypoints_loss(input, labels):
    loss = tf.reduce_mean(tf.nn.l2_loss(input - labels))
    return loss

