# -*- coding: utf-8 -*-
import tensorflow as tf
import gc
import re
from dataset import DataSet
from baseline import Baseline

class Baseline_axn2(Baseline):

    def __init__(self, ds):
        Baseline.__init__(self,ds, 'AXN2')

        # If a model is trained with multiple GPUs, prefix all Op names with tower_name
        # to differentiate the operations. Note that this prefix is removed from the
        # names of the summaries when visualizing a model.
        self.TOWER_NAME = 'tower'

    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.

        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('{}_[0-9]*/'.format(self.TOWER_NAME), '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        var = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var



    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
          total_loss: Total loss from loss().
        Returns:
          loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op


    def create(self,
               optimizer_params,
               batch_size = 128, lamb_reg = 0.0005, padding = 'SAME',
               stride = 2,
               l1filter = 1,
               l2filter = 1,
               l3filter = 1):

        self.params = {'batch_size': batch_size, 'lamb_reg': lamb_reg, 'padding' : padding, 'stride':stride,
                       'l1filter':l1filter, 'l2filter':l2filter, 'l3filter':l3filter }
        print('Parameters: ', self.params);

        self.optimizer_params = optimizer_params;
        print('Optimizer Parameters: ', self.optimizer_params.to_string());

        self._batch_size = batch_size

        num_labels = 10

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                                   shape=(batch_size, self._ds.image_size, self._ds.image_size, self._ds.num_channels),
                                                   name ='tf_train_dataset')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            with tf.device('/cpu:0'):
                if self._use_valid:
                    self.tf_valid_dataset = tf.constant(self._ds.valid_dataset)
                self.tf_test_dataset = tf.constant(self._ds.test_dataset)


            self.tf_keep_prob = tf.placeholder(tf.float32)
            kernel1 = self._variable_with_weight_decay('weights1',
                                                       shape=[5, 5, 3, 64],
                                                       stddev=5e-2,
                                                       wd=0.0)
            biases1 = tf.get_variable('biases1', [64], initializer=tf.constant_initializer(0.0))
            kernel2 = self._variable_with_weight_decay('weights2',
                                                       shape=[5, 5, 64, 64],
                                                       stddev=5e-2,
                                                       wd=0.0)
            biases2 = tf.get_variable('biases2', [64], initializer=tf.constant_initializer(0.1))
            weights3 = self._variable_with_weight_decay('weights3', shape=[4096, 384],
                                                        stddev=0.04, wd=0.004)
            biases3 = tf.get_variable('biases3', [384], initializer=tf.constant_initializer(0.1))
            weights4 = self._variable_with_weight_decay('weights4', shape=[384, 192],
                                                        stddev=0.04, wd=0.004)
            biases4 = tf.get_variable('biases4', [192], initializer=tf.constant_initializer(0.1))
            weights5 = self._variable_with_weight_decay('weights5', [192, num_labels],
                                                        stddev=1 / 192.0, wd=0.0)
            biases5 = tf.get_variable('biases5', [num_labels],
                                      initializer=tf.constant_initializer(0.0))

            # Model with dropout
            def model(data ):

                with tf.variable_scope('conv1') as scope:

                    conv1 = tf.nn.conv2d(data, kernel1, [1, 1, 1, 1], padding='SAME')
                    bias1 = tf.nn.bias_add(conv1, biases1)
                    conv1 = tf.nn.relu(bias1, name=scope.name)
                    self._activation_summary(conv1)

                # pool1
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool1')
                # norm1
                norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                  name='norm1')

                # conv2
                with tf.variable_scope('conv2') as scope:

                    conv2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')

                    bias2 = tf.nn.bias_add(conv2, biases2)
                    conv2 = tf.nn.relu(bias2, name=scope.name)
                    self._activation_summary(conv2)

                # norm2
                norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                  name='norm2')
                # pool2
                pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1], padding='SAME', name='pool2')

                # local3
                with tf.variable_scope('local3') as scope:
                    # Move everything into depth so we can perform a single matrix multiply.
                    shape = pool2.get_shape().as_list()
                    reshape3 = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
                    dim3 = reshape3.get_shape()[1].value
                    local3 = tf.nn.relu(tf.matmul(reshape3, weights3) + biases3, name=scope.name)
                    self._activation_summary(local3)

                # local4
                with tf.variable_scope('local4') as scope:

                    local4 = tf.nn.relu(tf.matmul(local3, weights4) + biases4, name=scope.name)
                    self._activation_summary(local4)

                # softmax, i.e. softmax(WX + b)
                with tf.variable_scope('softmax_linear') as scope:

                    softmax_linear = tf.add(tf.matmul(local4, weights5), biases5, name=scope.name)
                    self._activation_summary(softmax_linear)

                return softmax_linear

            logits = model(self.tf_train_dataset)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits, self.tf_train_labels, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            # The total loss is defined as the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)

            if self._use_valid:
                with tf.device('/cpu:0'): # Comment this out if you have a GPU > 8Gb
                    self.valid_prediction = tf.nn.softmax(model(self.tf_valid_dataset))

            # Run the test prediction on the CPU. This will keep the GPU memory under 8Gb
            # and doesn't affect the performance by much as it's only performed once at the end.
            with tf.device('/cpu:0'):
                self.test_prediction = tf.nn.softmax(model(self.tf_test_dataset))