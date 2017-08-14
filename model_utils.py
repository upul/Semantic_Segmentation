import tensorflow as tf
import numpy as np

vgg_weights = np.load('./pretrained_weights/vgg16_weights.npz')


def conv_layer(parent, kernel, bias, name):
    """
    This simple utility function create a convolution layer
    and applied relu activation.

    :param parent:
    :param kernel: Kernel weight tensor
    :param bias: Bias tensor
    :param name: Name of this layer
    :return: Convolution layer created according to the given parameters.
    """
    with tf.variable_scope(name) as scope:
        init = tf.constant_initializer(value=kernel, dtype=tf.float32)
        kernel = tf.get_variable(name="weights", initializer=init, shape=kernel.shape)
        conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')

        init = tf.constant_initializer(value=bias, dtype=tf.float32)
        biases = tf.get_variable(name="biases", initializer=init, shape=bias.shape)

        conv_with_bias = tf.nn.bias_add(conv, biases)
        conv_with_relu = tf.nn.relu(conv_with_bias, name=scope.name)
    return conv_with_relu


def max_pool_layer(parent, kernel, stride, name, padding='SAME'):
    max_pool = tf.nn.max_pool(parent, ksize=kernel, strides=stride, padding=padding, name=name)
    return max_pool


def fully_collected_layer(parent, name):
    with tf.variable_scope(name):
        if name == 'fc_1':
            pass

        if name == 'fc_2':
            pass

        if name == 'fc_3':
            pass

        raise RuntimeError('{} is not supported as a fully connected name'.format(name))