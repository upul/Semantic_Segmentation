import os

import shutil
import tensorflow as tf
import time
import numpy as np

import helper
from loss import logistic_loss
from model_utils import conv_layer
from model_utils import max_pool_layer
from model_utils import fully_collected_layer
from model_utils import upsample_layer
from model_utils import skip_layer_connection
from model_utils import preprocess


class FCN:
    def __init__(self, input_shape, num_train_examples, viz_dir, batch_size=8, num_classes=2):
        self.images_batch, self.labels_batch, self.images_viz, self.dropout, self.global_step = self._build_placeholders(
            input_shape)
        self.input_shape = input_shape
        self.viz_dir = viz_dir
        self.num_train_examples = num_train_examples
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.sess = tf.Session()
        self.logits = self.build()

    def build(self):
        images = preprocess(self.images_batch)

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
        img_shape = tf.shape(self.images_batch)
        dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.n_classes])
        upsample_1 = upsample_layer(fc_3, dconv3_shape, self.n_classes, 'upsample_1', 32)

        skip_1 = skip_layer_connection(pool4, 'skip_1', 512, stddev=0.00001)
        upsample_2 = upsample_layer(skip_1, dconv3_shape, self.n_classes, 'upsample_2', 16)

        skip_2 = skip_layer_connection(pool3, 'skip_2', 256, stddev=0.0001)
        upsample_3 = upsample_layer(skip_2, dconv3_shape, self.n_classes, 'upsample_3', 8)

        logit = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
        return logit

    def optimize(self, batch_generator, learning_rate=1e-5, keep_prob=0.75, num_epochs=1):
        loss = logistic_loss(logits=self.logits, labels=self.labels_batch, n_classes=self.n_classes)
        summary_op = self._build_summary(loss=loss)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

        validation_img_summary_op = tf.summary.image('validation_img', self.images_viz)

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # to visualize using TensorBoard
        writer = tf.summary.FileWriter('./graphs/kitti/', self.sess.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/kitti/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print('Graph is available in hard disk. Hence, loading it.')
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        initial_step = self.sess.run(self.global_step)
        num_batches = int(self.num_train_examples / self.batch_size)
        for itr in range(initial_step, num_batches * num_epochs):
            image, gt_image = next(batch_generator(self.batch_size))
            _, loss_val, summary = self.sess.run([optimizer, loss, summary_op],
                                                 feed_dict={self.images_batch: image,
                                                            self.labels_batch: gt_image,
                                                            self.dropout: keep_prob})
            writer.add_summary(summary, global_step=itr)

            if (itr < 10) or (itr < 100 and itr % 10 == 0) or \
                    (itr < 1000 and itr % 100 == 0) or (itr >= 1000 and itr % 200 == 0):
                epoch_no = int(itr / num_batches)
                print('epoch: {0:>3d}  iter: {1:>4d}  loss: {2:>8.4e}'.format(epoch_no, itr, loss_val))

            if itr % 10 == 0:
                viz_images = self.training_visulize()
                tt = self.sess.run([validation_img_summary_op], feed_dict={self.images_viz: viz_images})
                writer.add_summary(tt[0], itr)

            if ((itr + 1) % (num_batches * 20) == 0) or (itr == num_batches * num_epochs):
                print('At iteration: {} save a checkpoint'.format(itr))
                saver.save(self.sess, './checkpoints/kitti/state', itr)

    def inference(self, runs_dirs, data_dirs):
        reshape_logits = tf.reshape(self.logits, (-1, self.n_classes))
        helper.save_inference_samples(runs_dirs, data_dirs, self.sess, self.input_shape, reshape_logits, self.dropout,
                                      self.images_batch)

    def close_session(self):
        self.sess.close()

    def training_visulize(self):
        reshape_logits = tf.reshape(self.logits, (-1, self.n_classes))
        viz_images = []
        img_output = helper.gen_test_output(self.sess, reshape_logits, self.dropout, self.images_batch, self.viz_dir,
                                            self.input_shape)
        for _, ouput in img_output:
            viz_images.append(ouput)
        return np.array(viz_images)

    @staticmethod
    def _build_placeholders(shape):
        with tf.name_scope('data'):
            X = tf.placeholder(tf.float32, [None, shape[0], shape[1], 3], name='X_placeholder')
            Y = tf.placeholder(tf.float32, [None, shape[0], shape[1], 2], name='Y_placeholder')
            X_viz = tf.placeholder(tf.float32, [None, shape[0], shape[1], 3], name='X_valid_placeholder')

        dropout = tf.placeholder(tf.float32, name='dropout')
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        return X, Y, X_viz, dropout, global_step

    @staticmethod
    def _build_summary(loss):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('histogram loss', loss)
            return tf.summary.merge_all()


if __name__ == '__main__':
    num_classes = 2
    images_per_batch = 8
    data_dir = './data'
    runs_dir = './runs'
    viz_dir = './data/data_road/validating'
    input_size = (160, 576)
    num_train_examples = 289

    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), input_size)
    fc_network = FCN(input_size, num_train_examples, viz_dir, images_per_batch, num_classes)
    fc_network.optimize(get_batches_fn, num_epochs=25)
    fc_network.inference(runs_dir, data_dir)
