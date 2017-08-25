import tensorflow as tf

from loss import logistic_loss
from model_utils import preprocess, conv_layer, max_pool_layer, fully_collected_layer, upsample_layer, \
    skip_layer_connection


class FCN:
    def __init__(self, img_placeholder, batch_size, num_classes=2):
        self.img_placeholder = img_placeholder
        self.batch_size = batch_size
        self.num_classes = num_classes

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

        fc_1 = fully_collected_layer(pool5, 'fc_1')
        fc_2 = fully_collected_layer(fc_1, 'fc_2')
        fc_3 = fully_collected_layer(fc_2, 'fc_3')

        # New we start upsampling and skip layer connections.
        img_shape = tf.shape(self.img_placeholder)
        dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.num_classes])
        upsample_1 = upsample_layer(fc_3, dconv3_shape, self.num_classes, 'upsample_1', 32)

        skip_1 = skip_layer_connection(pool4, 'skip_1', 512)
        upsample_2 = upsample_layer(skip_1, dconv3_shape, self.num_classes, 'upsample_2', 16)

        skip_2 = skip_layer_connection(pool3, dconv3_shape, 'skip_2', 256)
        upsample_3 = upsample_layer(skip_2, dconv3_shape, self.num_classes, 'upsample_3', 8)

        logit = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
        return logit

    def optimize(self, logits, label_placeholder, num_classes=2):
        loss = logistic_loss(logits=logits, labels=label_placeholder, num_classes=num_classes)




    def inference(self):
        pass
