# -*- coding: utf-8 -*-
import tensorflow as tf
from dataset import DataSet
from baseline import Baseline

class Baseline_cnn(Baseline):

    def __init__(self, ds):
        Baseline.__init__(self,ds, 'CNN')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        print('initial ', initial)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create(self, batch_size = 16,
                     patch_size = 5,
                     depth1 = 32,
                     depth2 = 64,
                     num_hidden = 1024,
                     pooling_ksize = [1, 2, 2, 1],
                     pooling_strides = [1, 2, 2, 1]):

        print('Parameters: batch_size: {}, patch_size: {}, depth1: {}, depth2: {}, num_hidden: {}, pooling_ksize: {}, pooling_strides: {}'
              .format(batch_size, patch_size, depth1, depth2, num_hidden, pooling_ksize, pooling_strides))

        image_size = 32
        num_labels = 10
        num_channels = 3  # RGB

        self._batch_size = batch_size

        self.graph = tf.Graph()

        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            self.tf_test_dataset = tf.constant(self._ds.test_dataset)

            # Variables.
            layer1_weights = self.weight_variable([patch_size, patch_size, num_channels, depth1])
            layer1_biases = self.bias_variable([depth1])
            layer2_weights = self.weight_variable([patch_size, patch_size, depth1, depth2])
            layer2_biases = self.bias_variable([depth2])
            layer3_weights = self.weight_variable([image_size // 4 * image_size // 4 * depth2, num_hidden])
            layer3_biases = self.bias_variable([num_hidden])
            layer4_weights = self.weight_variable([num_hidden, num_labels])
            layer4_biases = self.bias_variable([num_labels])

            self.tf_keep_prob = tf.placeholder("float")

            # Model with dropout
            def model(data, proba):
                # Convolution
                conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases
                pooled1 = tf.nn.max_pool(tf.nn.relu(conv1), ksize=pooling_ksize,
                                         strides=pooling_strides, padding='SAME')
                # Convolution
                conv2 = tf.nn.conv2d(pooled1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases
                pooled2 = tf.nn.max_pool(tf.nn.relu(conv2), ksize=pooling_ksize,
                                         strides=pooling_strides, padding='SAME')
                # Fully Connected Layer
                shape = pooled2.get_shape().as_list()
                reshape = tf.reshape(pooled2, [shape[0], shape[1] * shape[2] * shape[3]])
                full3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                # Dropout
                full3 = tf.nn.dropout(full3, proba)
                return tf.matmul(full3, layer4_weights) + layer4_biases

            # Training computation.
            logits = model(self.tf_train_dataset, self.tf_keep_prob)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))

            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            self.test_prediction = tf.nn.softmax(model(self.tf_test_dataset, 1.0))

