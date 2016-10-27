# -*- coding: utf-8 -*-
import tensorflow as tf
import gc
from dataset import DataSet
from baseline import Baseline

class Baseline_axn(Baseline):

    def __init__(self, ds):
        Baseline.__init__(self,ds, 'AXN')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        print('initial ', initial)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create(self, batch_size = 64, lamb_reg = 0.0005, padding = 'SAME',
               stride = 2,
               l1filter = 1,
               l2filter = 1,
               l3filter = 1):

        self.params = {'batch_size': batch_size, 'lamb_reg': lamb_reg, 'padding' : padding, 'stride':stride,
                       'l1filter':l1filter, 'l2filter':l2filter, 'l3filter':l3filter }
        print('Parameters: ', self.params);

        self._batch_size = batch_size

        num_labels = 10

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, self._ds.image_size, self._ds.image_size, self._ds.num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            with tf.device('/cpu:0'):
                self.tf_valid_dataset = tf.constant(self._ds.valid_dataset)
                self.tf_test_dataset = tf.constant(self._ds.test_dataset)


            # Variables.
            num_channels=3
            layer1_weights = self.weight_variable([3, 3, num_channels, 64])
            layer1_biases = self.bias_variable([64])
            layer2_weights = self.weight_variable([3, 3, 64, 128])
            layer2_biases = self.bias_variable([128])
            layer3_weights = self.weight_variable([3, 3, 128, 256])
            layer3_biases = self.bias_variable([256])
            layer4_weights = self.weight_variable([4 * 4 * 256, 1024])
            layer4_biases = self.bias_variable([1024])
            layer5_weights = self.weight_variable([1024, 1024])
            layer5_biases = self.bias_variable([1024])
            layer6_weights = self.weight_variable([1024, num_labels])
            layer6_biases = self.bias_variable([num_labels])

            self.tf_keep_prob = tf.placeholder(tf.float32)

            # Model with dropout
            def model(data, proba=self.tf_keep_prob):

                # Convolution
                conv1 = tf.nn.conv2d(data, layer1_weights, [l1filter, l1filter, 1, 1], padding=padding) + layer1_biases
                # Max pooling
                pooled1 = tf.nn.max_pool(tf.nn.relu(conv1), ksize=[1, 3, 3, 1],
                                         strides=[1, stride, stride, 1], padding=padding)
                # Normalization
                norm1 = tf.nn.lrn(pooled1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                # Dropout
                norm1 = tf.nn.dropout(norm1, proba)

                # Convolution
                conv2 = tf.nn.conv2d(norm1, layer2_weights, [l2filter, l2filter, 1, 1], padding=padding) + layer2_biases
                # Max pooling
                pooled2 = tf.nn.max_pool(tf.nn.relu(conv2), ksize=[1, 3, 3, 1],
                                         strides=[1, stride, stride, 1], padding=padding)
                # Normalization
                norm2 = tf.nn.lrn(pooled2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                # Dropout
                norm2 = tf.nn.dropout(norm2, proba)

                # Convolution
                conv3 = tf.nn.conv2d(norm2, layer3_weights, [l3filter, l3filter, 1, 1], padding=padding) + layer3_biases
                # Max pooling
                pooled3 = tf.nn.max_pool(tf.nn.relu(conv3), ksize=[1, 3, 3, 1],
                                         strides=[1, stride, stride, 1], padding=padding)
                # Normalization
                norm3 = tf.nn.lrn(pooled3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                # Dropout
                norm3 = tf.nn.dropout(norm3, proba)

                # Fully Connected Layer
                shape = layer4_weights.get_shape().as_list()
                reshape = tf.reshape(norm3, [-1, shape[0]])
                full3 = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
                full3 = tf.nn.relu(tf.matmul(full3, layer5_weights) + layer5_biases)

                return tf.matmul(full3, layer6_weights) + layer6_biases

            # Training computation.
            logits = model(self.tf_train_dataset, self.tf_keep_prob)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
            regularizers = (tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) +
                            tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) +
                            tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) +
                            tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases) +
                            tf.nn.l2_loss(layer5_weights) + tf.nn.l2_loss(layer5_biases) +
                            tf.nn.l2_loss(layer6_weights) + tf.nn.l2_loss(layer6_biases))

            # Add the regularization term to the loss.
            self.loss = tf.reduce_mean(loss + lamb_reg * regularizers)
            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)


            with tf.device('/cpu:0'): # Comment this out if you have a GPU > 8Gb
                self.valid_prediction = tf.nn.softmax(model(self.tf_valid_dataset, 1.0))

            with tf.device('/cpu:0'):
                self.test_prediction = tf.nn.softmax(model(self.tf_test_dataset, 1.0))