import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    xt = tf.Variable([1.0, 1.0], [2.0, 2.0])
    xx1 = tf.Variable([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    xx2 = tf.Variable([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    xx = tf.Variable([[1.0, 2.0], [3.0, 4.0]])

    aa = tf.Variable([[[1.0, 2.2, 2.2, 3.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    aa_shape = tf.shape(aa)
    aa_min, indices = tf.math.top_k(aa, k=2, sorted=True, name=None)
    # aa_min_shape = tf.shape(aa_min)


    a1 = tf.expand_dims(xx1, axis=1)
    a2 = tf.expand_dims(xx2, axis=0)

    sess = tf.InteractiveSession()

    xt.initializer.run()
    xx.initializer.run()
    xx1.initializer.run()
    xx2.initializer.run()
    aa.initializer.run()

    # print(sess.run(a1-a2))
    #
    # shape = tf.shape(a1 - a2)
    # print(sess.run(shape))

    print(sess.run(aa))
    print(sess.run(aa_shape))
    print(sess.run(aa_min))
    # print(sess.run(aa_min_shape))


    return 0


if __name__ == '__main__':
    main()