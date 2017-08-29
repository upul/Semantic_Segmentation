import tensorflow as tf


def logistic_loss(logits, labels, n_classes):
    with tf.variable_scope('logistic_loss'):
        reshaped_logits = tf.reshape(logits, (-1, n_classes))
        reshaped_labels = tf.reshape(labels, (-1, n_classes))
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=reshaped_logits,
                                                          labels=reshaped_labels)
        loss = tf.reduce_mean(entropy, name='logistic_loss')
        return loss
