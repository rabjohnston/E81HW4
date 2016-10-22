# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from six.moves import cPickle as pickle
from six.moves import range

import tensorflow as tf

import DataSet

class Baseline:

    def __init__(self, ds):
        self._ds = ds

    def weight_variable(self, shape):
        # initial = tf.truncated_normal(shape, stddev=0.01)
        initial = tf.truncated_normal(shape, stddev=tf.sqrt(2.0 / shape[0]))
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # initial = tf.constant(0.1, shape=shape)
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def run(self):
        
        batch_size = 256
        hidden_nodes = 1024

        split_by_half = lambda x, k: int(x / 2 ** k)

        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 784))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))
            tf_valid_dataset = tf.constant(self._ds.valid_dataset)
            tf_test_dataset = tf.constant(self._ds.test_dataset)

            # Variables.
            layer1_weights = self.weight_variable([784, hidden_nodes])
            layer1_biases = self.bias_variable([hidden_nodes])
            layer2_weights = self.weight_variable([hidden_nodes, split_by_half(hidden_nodes, 1)])
            layer2_biases = self.bias_variable([split_by_half(hidden_nodes, 1)])
            layer3_weights = self.weight_variable([split_by_half(hidden_nodes, 1), split_by_half(hidden_nodes, 2)])
            layer3_biases = self.bias_variable([split_by_half(hidden_nodes, 2)])
            layer4_weights = self.weight_variable([split_by_half(hidden_nodes, 2), 10])
            layer4_biases = self.bias_variable([10])

            keep_prob = tf.placeholder("float")

            # Model with dropout
            def model(data, proba=keep_prob):
                layer1 = tf.matmul(data, layer1_weights) + layer1_biases
                hidden1 = tf.nn.dropout(tf.nn.relu(layer1), proba)  # dropout on hidden layer
                layer2 = tf.matmul(hidden1, layer2_weights) + layer2_biases  # a new hidden layer
                hidden2 = tf.nn.dropout(tf.nn.relu(layer2), proba)
                layer3 = tf.matmul(hidden2, layer3_weights) + layer3_biases
                hidden3 = tf.nn.dropout(tf.nn.relu(layer3), proba)
                return tf.matmul(hidden3, layer4_weights) + layer4_biases

            # Training computation.
            logits = model(tf_train_dataset, keep_prob)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
            regularizers = (tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) + \
                            tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) + \
                            tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + \
                            tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases))

            # Add the regularization term to the loss.
            # loss += lamb_reg * regularizers
            loss = tf.reduce_mean(loss + lamb_reg * regularizers)

            # Optimizer.
            # learning rate decay
            global_step = tf.Variable(0)  # count  number of steps taken.
            start_learning_rate = 0.5
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
            test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))




def main():
    d = DataSet()
    d.load()

    # Visualise the data for one image
    #d.display(40000)

    print('Finished')
    
if __name__ == '__main__':
  main()
  
  
    