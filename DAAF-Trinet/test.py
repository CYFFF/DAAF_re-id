# import tensorflow as tf
import scipy.io as sio
import numpy as np
import os

def main():

    # g1 = tf.Graph()
    # with g1.as_default():
    #     a1 = tf.get_variable("a1", initializer=tf.zeros_initializer(shape=[1]))
    #
    # a = tf.constant([1.0, 2.0], name="a")
    # b = tf.constant([2.0, 3.0], name="b")
    #
    # c = a + b
    # print(c)

    # a = sio.loadmat('attribute/market_attribute.mat')
    # b = a['market_attribute']['train'][0, 0]
    # d = b['age'][0, 0]
    # e = d[0, 7]
    # c = h5py.File('attribute/market_attribute.mat')
    


    image_root = '/data/chenyifan/Market-1501/bounding_box_train'
    images = os.listdir(image_root)
    name_dict = {}
    index = 0
    images.sort()
    for image in images:
        name = image[0:4]
        if not name in name_dict:
            name_dict[name] = index
            index = index + 1
    
    a = sio.loadmat('attribute/market_attribute.mat')
    aaaaaaaaa = a['market_attribute']['train'][0, 0]

    for image in images:
        name = image[0:4]
        relative_id = name_dict[name]

        aaaaaaaaa['age'][0, 0][0, relative_id]
        aaaaaaaaa['backpack'][0, 0][0, relative_id]
        aaaaaaaaa['bag'][0, 0][0, relative_id]
        aaaaaaaaa['handbag'][0, 0][0, relative_id]
        aaaaaaaaa['downblack'][0, 0][0, relative_id]
        aaaaaaaaa['downblue'][0, 0][0, relative_id]
        aaaaaaaaa['downbrown'][0, 0][0, relative_id]
        aaaaaaaaa['downgray'][0, 0][0, relative_id]
        aaaaaaaaa['downgreen'][0, 0][0, relative_id]
        aaaaaaaaa['downpink'][0, 0][0, relative_id]
        aaaaaaaaa['downpurple'][0, 0][0, relative_id]
        aaaaaaaaa['downwhite'][0, 0][0, relative_id]
        aaaaaaaaa['downyellow'][0, 0][0, relative_id]
        aaaaaaaaa['upblack'][0, 0][0, relative_id]
        aaaaaaaaa['upblue'][0, 0][0, relative_id]
        aaaaaaaaa['upgreen'][0, 0][0, relative_id]
        aaaaaaaaa['upgray'][0, 0][0, relative_id]
        aaaaaaaaa['uppurple'][0, 0][0, relative_id]
        aaaaaaaaa['upred'][0, 0][0, relative_id]
        aaaaaaaaa['upwhite'][0, 0][0, relative_id]
        aaaaaaaaa['upyellow'][0, 0][0, relative_id]
        aaaaaaaaa['clothes'][0, 0][0, relative_id]
        aaaaaaaaa['down'][0, 0][0, relative_id]
        aaaaaaaaa['up'][0, 0][0, relative_id]
        aaaaaaaaa['hair'][0, 0][0, relative_id]
        aaaaaaaaa['hat'][0, 0][0, relative_id]
        aaaaaaaaa['gender'][0, 0][0, relative_id]

    print('111')


if __name__ == '__main__':
    main()

