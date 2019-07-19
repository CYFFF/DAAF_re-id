import tensorflow as tf

def main():

    g1 = tf.Graph()
    with g1.as_default():
        a1 = tf.get_variable("a1", initializer=tf.zeros_initializer(shape=[1]))

    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")

    c = a + b

    print(c)



if __name__ == '__main__':
    main()

