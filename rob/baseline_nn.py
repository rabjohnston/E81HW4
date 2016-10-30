# -*- coding: utf-8 -*-



import tensorflow as tf

from dataset import DataSet
from baseline import Baseline


class Baseline_nn(Baseline):
    """
    Basic three layer NN with ability to tune a number of hyper-parameters
    """
    def __init__(self, ds):
        Baseline.__init__(self,ds,'NN')


    def weight_variable(self, shape):
        # initial = tf.truncated_normal(shape, stddev=0.01)
        initial = tf.truncated_normal(shape, stddev=tf.sqrt(2.0 / shape[0]))
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # initial = tf.constant(0.1, shape=shape)
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def create(self, batch_size = 128, hidden_nodes = 1024, lamb_reg=0.01, start_learning_rate = 0.01):

        # Save the hyper params for later
        self.params = {'batch_size':batch_size, 'hidden_nodes':hidden_nodes, 'lamb_reg':lamb_reg,
                       'start_learning_rate':start_learning_rate }
        self._batch_size = batch_size

        print('Parameters: batch_size: {}, hidden_nodes: {}, lamb_reg: {}, start_learning_rate: {}'
              .format(batch_size, hidden_nodes, lamb_reg, start_learning_rate))

        split_by_half = lambda x, k: int(x / 2 ** k)

        input_layer_size = self._ds.image_size * self._ds.image_size * self._ds.num_channels
        output_layer_size = 10

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_layer_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_layer_size))
            self.tf_test_dataset = tf.constant(self._ds.test_dataset)

            # Variables.
            layer1_weights = self.weight_variable([input_layer_size, hidden_nodes])
            layer1_biases = self.bias_variable([hidden_nodes])
            layer2_weights = self.weight_variable([hidden_nodes, split_by_half(hidden_nodes, 1)])
            layer2_biases = self.bias_variable([split_by_half(hidden_nodes, 1)])
            layer3_weights = self.weight_variable([split_by_half(hidden_nodes, 1), split_by_half(hidden_nodes, 2)])
            layer3_biases = self.bias_variable([split_by_half(hidden_nodes, 2)])
            layer4_weights = self.weight_variable([split_by_half(hidden_nodes, 2), output_layer_size])
            layer4_biases = self.bias_variable([output_layer_size])

            self.tf_keep_prob = tf.placeholder("float")

            # Model with dropout
            def model(data, proba=self.tf_keep_prob):
                layer1 = tf.matmul(data, layer1_weights) + layer1_biases
                hidden1 = tf.nn.dropout(tf.nn.relu(layer1), proba)  # dropout on hidden layer
                layer2 = tf.matmul(hidden1, layer2_weights) + layer2_biases  # a new hidden layer
                hidden2 = tf.nn.dropout(tf.nn.relu(layer2), proba)
                layer3 = tf.matmul(hidden2, layer3_weights) + layer3_biases
                hidden3 = tf.nn.dropout(tf.nn.relu(layer3), proba)
                return tf.matmul(hidden3, layer4_weights) + layer4_biases

            # Training computation.
            logits = model(self.tf_train_dataset, self.tf_keep_prob)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
            regularizers = (tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) + \
                            tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) + \
                            tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + \
                            tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases))

            # Add the regularization term to the loss.
            # loss += lamb_reg * regularizers
            self.loss = tf.reduce_mean(loss + lamb_reg * regularizers)

            # Optimizer.
            # learning rate decay
            self.global_step = tf.Variable(0)  # count  number of steps taken.

            learning_rate = tf.train.exponential_decay(start_learning_rate, self.global_step, 100000, 0.96, staircase=True)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            self.test_prediction = tf.nn.softmax(model(self.tf_test_dataset, 1.0))





  
  
    