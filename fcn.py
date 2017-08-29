import os

import shutil
import tensorflow as tf
import time

import helper
from loss import logistic_loss
from model_utils import conv_layer
from model_utils import max_pool_layer
from model_utils import fully_collected_layer
from model_utils import upsample_layer
from model_utils import skip_layer_connection
from model_utils import preprocess


class FCN:
    def __init__(self, input_shape, batch_size=8, num_classes=2):
        self.input_shape = input_shape
        X, Y, dropout = self._build_placeholders(input_shape)
        self.img_placeholder = X
        self.lbl_placeholder = Y
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.sess = tf.Session()
        self.logits = self.build()

    def build(self):
        images = preprocess(self.img_placeholder)

        conv1_1 = conv_layer(images, 'conv1_1_W', 'conv1_1_b', name='conv1_1')
        conv1_2 = conv_layer(conv1_1, 'conv1_2_W', 'conv1_2_b', name='conv1_2')
        pool1 = max_pool_layer(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool1')

        conv2_1 = conv_layer(pool1, 'conv2_1_W', 'conv2_1_b', name='conv2_1')
        conv2_2 = conv_layer(conv2_1, 'conv2_2_W', 'conv2_2_b', name='conv2_2')
        pool2 = max_pool_layer(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool2')

        conv3_1 = conv_layer(pool2, 'conv3_1_W', 'conv3_1_b', name='conv3_1')
        conv3_2 = conv_layer(conv3_1, 'conv3_2_W', 'conv3_2_b', name='conv3_2')
        conv3_3 = conv_layer(conv3_2, 'conv3_3_W', 'conv3_3_b', name='conv3_3')
        pool3 = max_pool_layer(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool3')

        conv4_1 = conv_layer(pool3, 'conv4_1_W', 'conv4_1_b', name='conv4_1')
        conv4_2 = conv_layer(conv4_1, 'conv4_2_W', 'conv4_2_b', name='conv4_2')
        conv4_3 = conv_layer(conv4_2, 'conv4_3_W', 'conv4_3_b', name='conv4_3')
        pool4 = max_pool_layer(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool4')

        conv5_1 = conv_layer(pool4, 'conv5_1_W', 'conv5_1_b', name='conv5_1')
        conv5_2 = conv_layer(conv5_1, 'conv5_2_W', 'conv5_2_b', name='conv5_2')
        conv5_3 = conv_layer(conv5_2, 'conv5_3_W', 'conv5_3_b', name='conv5_3')
        pool5 = max_pool_layer(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool5')

        fc_1 = fully_collected_layer(pool5, 'fc_1', self.dropout)
        fc_2 = fully_collected_layer(fc_1, 'fc_2', self.dropout)
        fc_3 = fully_collected_layer(fc_2, 'fc_3', self.dropout)

        # New we start upsampling and skip layer connections.
        img_shape = tf.shape(self.img_placeholder)
        dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.n_classes])
        upsample_1 = upsample_layer(fc_3, dconv3_shape, self.n_classes, 'upsample_1', 32)

        skip_1 = skip_layer_connection(pool4, 'skip_1', 512)
        upsample_2 = upsample_layer(skip_1, dconv3_shape, self.n_classes, 'upsample_2', 16)

        skip_2 = skip_layer_connection(pool3, 'skip_2', 256)
        upsample_3 = upsample_layer(skip_2, dconv3_shape, self.n_classes, 'upsample_3', 8)

        logit = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
        return logit

    def optimize(self, batch_generator, learning_rate=1e-5, keep_prob=0.75, num_epochs=1):
        loss = logistic_loss(logits=self.logits, labels=self.lbl_placeholder, n_classes=self.n_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            itr = 0
            for image, gt_image in batch_generator(self.batch_size):
                _, loss_val = self.sess.run([optimizer, loss],
                                            feed_dict={self.img_placeholder: image,
                                                       self.lbl_placeholder: gt_image,
                                                       self.dropout: keep_prob})
                if (itr < 10) or (itr < 100 and itr % 10 == 0) or \
                        (itr < 1000 and itr % 100 == 0) or (itr >= 1000 and itr % 200 == 0):
                    print('epoch: {0:>3d}  iter: {1:>4d}  loss: {2:>8.4e}'.format(epoch, itr, loss_val))
                itr += 1

    def inference(self, runs_dirs, data_dirs):
        reshape_logits = tf.reshape(self.logits, (-1, self.n_classes))
        helper.save_inference_samples(runs_dirs, data_dirs, self.sess, self.input_shape, reshape_logits, self.dropout,
                                      self.img_placeholder)

    def close_session(self):
        self.sess.close()

    @staticmethod
    def _build_placeholders(shape):
        with tf.name_scope('data'):
            X = tf.placeholder(tf.float32, [None, shape[0], shape[1], 3], name='X_placeholder')
            Y = tf.placeholder(tf.float32, [None, shape[0], shape[1], 2], name='Y_placeholder')
        dropout = tf.placeholder(tf.float32, name='dropout')
        return X, Y, dropout


if __name__ == '__main__':
    num_classes = 2
    images_per_batch = 4
    data_dir = './data'
    runs_dir = './runs'
    input_size = (160, 576)

    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), input_size)
    fc_network = FCN(input_size, images_per_batch, num_classes)
    fc_network.optimize(get_batches_fn)
    fc_network.inference(runs_dir, data_dir)
