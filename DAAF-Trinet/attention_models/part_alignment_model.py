import tensorflow.compat.v1 as tf
import tensorflow as tf_v2
from nets import resnet_utils
import tf_slim as slim
from nets.resnet_v1 import resnet_v1_block, resnet_v1


# from nets.resnet_v1 import resnet_v1_50_block_4, resnet_v1_50_block_3_4, resnet_v1_50_block_2_3_4, resnet_arg_scope
from nets.resnet_v1 import resnet_arg_scope
_RGB_MEAN = [123.68, 116.78, 103.94]


# def resnet_block_4(inputs, is_training=True):
#     with slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
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
#     with slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
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
#     with slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
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
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='trans_conv_0')
    return net


def trans_conv_1(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_1', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='trans_conv_1')
    return net


def trans_conv_2(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_2', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', scope='trans_conv_2')
    return net


def trans_conv_3(inputs, kp_num=1):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_3', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d_transpose(inputs, 64, [3, 3], stride=2, padding='SAME', activation_fn=None,
                                        scope='trans_conv_3')
    return net


def conv_0(inputs, kp_num=5):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_0'):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_0')
    return net

def conv_1(inputs, kp_num=5):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_1'):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_1')
    return net

def conv_2(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_2'):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_2')
    return net

def conv_3(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_3'):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_3')
    return net

def conv_4(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_4'):
        with slim.arg_scope(resnet_arg_scope()):
            net = slim.conv2d(inputs, kp_num, [1, 1], 1, 'SAME', activation_fn=tf.sigmoid, scope='conv1x1_4')
    return net

def conv_5(inputs, kp_num=4):
    depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope('keypoints_trans_conv_4_5'):
        with slim.arg_scope(resnet_arg_scope()):
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


def resnet_v1_50_block_5(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=tf.AUTO_REUSE,
                 scope='Resnet_block_5'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block5', base_depth=340, num_units=3, stride=1)
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=False, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v1_50_block_5.default_image_size = resnet_v1.default_image_size


def attribute_fc(input):
    fc = slim.fully_connected(
        input, 512, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        },
        weights_initializer=tf.orthogonal_initializer(),
        reuse=tf.AUTO_REUSE,
        scope='attribute_fc'
    )
    return fc


def attr_conv_fc_loss_0(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 4, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_age')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_gender')

    label_age_onehot = tf_v2.one_hot(ground_truth[:, 0], 4)
    label_gender_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_age_onehot) +\
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_gender_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_1(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_backpack')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_bag')
    result_2 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_handbag')

    label_backpack_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_bag_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)
    label_handbag_onehot = tf_v2.one_hot(ground_truth[:, 2], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_backpack_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_bag_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_2, label_handbag_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_2(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_hair')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_hat')

    label_hair_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_hat_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_hair_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_hat_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_3(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downblack')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downblue')
    result_2 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downbrown')
    result_3 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downgray')

    label_downblack_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_downblue_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)
    label_downbrown_onehot = tf_v2.one_hot(ground_truth[:, 2], 2)
    label_downgray_onehot = tf_v2.one_hot(ground_truth[:, 3], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_downblack_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_downblue_onehot)  + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_2, label_downbrown_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_3, label_downgray_onehot)

    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_4(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downgreen')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downpink')
    result_2 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downpurple')
    result_3 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downwhite')
    result_4 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_downyellow')

    label_downgreen_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_downpink_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)
    label_downpurple_onehot = tf_v2.one_hot(ground_truth[:, 2], 2)
    label_downwhite_onehot = tf_v2.one_hot(ground_truth[:, 3], 2)
    label_downyellow_onehot = tf_v2.one_hot(ground_truth[:, 4], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_downgreen_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_downpink_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_2, label_downpurple_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_3, label_downwhite_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_4, label_downyellow_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_5(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upblack')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upblue')
    result_2 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upgreen')
    result_3 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upgray')

    label_upblack_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_upblue_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)
    label_upgreen_onehot = tf_v2.one_hot(ground_truth[:, 2], 2)
    label_upgray_onehot = tf_v2.one_hot(ground_truth[:, 3], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_upblack_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_upblue_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_2, label_upgreen_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_3, label_upgray_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_6(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_uppurple')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upred')
    result_2 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upwhite')
    result_3 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_upyellow')

    label_uppurple_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_upred_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)
    label_upwhite_onehot = tf_v2.one_hot(ground_truth[:, 2], 2)
    label_upyellow_onehot = tf_v2.one_hot(ground_truth[:, 3], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_uppurple_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_upred_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_2, label_upwhite_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_3, label_upyellow_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean


def attr_conv_fc_loss_7(input, ground_truth, attr_num=2):
    _, out_0 = resnet_v1_50_block_5(input)
    out_1 = tf.reduce_mean(out_0['Resnet_block_5/block5'], [1, 2], name='pool6', keep_dims=False)
    out_2 = attribute_fc(out_1)

    result_0 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_clothes')
    result_1 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_down')
    result_2 = slim.fully_connected(
        out_2, 2, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='fc_up')

    label_clothes_onehot = tf_v2.one_hot(ground_truth[:, 0], 2)
    label_down_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)
    label_up_onehot = tf_v2.one_hot(ground_truth[:, 1], 2)

    loss = tf_v2.nn.softmax_cross_entropy_with_logits(result_0, label_clothes_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_1, label_down_onehot) + \
           tf_v2.nn.softmax_cross_entropy_with_logits(result_2, label_up_onehot)
    loss_mean = tf.reduce_mean(loss, keep_dims=False)

    return loss_mean