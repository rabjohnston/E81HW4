# -*- coding: utf-8 -*-

from dataset import DataSet
import tensorflow as tf
import numpy as np


class EpochValue:
    """
    Holds a snapshot view of the accuracy and loss at specific position in training
    """
    def __init__(self, batch, train_accuracy, valid_accuracy, loss):

        # The current batch number
        self.batch = batch

        # The current training accuraccy
        self.train_accuracy = train_accuracy

        # The current validation accuracy (may be None if we aren't using validation dataset)
        self.test_accuracy = valid_accuracy

        # The current loss value
        self.loss = loss


class Baseline:
    """
    Model base for NNs and CNNs. This allows us to build models and run them through tensorflow
    """
    def __init__(self, ds, name):

        # The cifar-10 datasets
        self._ds = ds

        # The name of the model
        self.name = name

        # Are we using a validation test data set? If so then we'll split off some data
        # for validation otherwise we'll use it all for training.
        self._use_valid = ds._use_valid

        # The batch size we're using
        self._batch_size = 0

        # Tensor flow variables for the training and test data
        self.tf_train_dataset = None
        self.tf_train_labels = None
        self.tf_valid_dataset = None
        self.tf_test_dataset = None

        # Tensor flow variable
        self.tf_keep_prob = None

        # The optimiser function
        self.optimizer = None

        # The loss function
        self.loss = None

        self.graph = None

        # The hyper parameters. These are stored in a dict so we can output them to file after a run.
        # This will be useful when we're evaluating the model
        self.params = None

        # The optimizers hyper-parameters. As per params above, we store them for later analysis
        self.optimizer_params = None

        # After training we'll save the predictions on the training and test data
        self.train_prediction = None
        self.valid_prediction = None
        self.test_prediction = None

        # After training we'll also store the test predicitons
        self.test_preds = None

        # Store info about how well we're doing each batch
        self.epochs = {}


    def accuracy(self, predictions, labels):
        """
        Defines an accuracy function
        """
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])


    def run_session(self, num_batches, k_prob=1.0):
        """
        Train the model and give a prediction based on the test data set.
        :param num_batches: The number of batches to run
        :param k_prob:
        """

        with tf.Session(graph = self.graph) as session:

            # Build the summary operation based on the TF collection of Summaries.
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", session.graph)

            tf.initialize_all_variables().run()
            print("Initialized")
            for batch in range(num_batches):
                offset = (batch * self._batch_size) % (self._ds.train_labels.shape[0] - self._batch_size)
                batch_data = self._ds.train_dataset[offset:(offset + self._batch_size), :]
                batch_labels = self._ds.train_labels[offset:(offset + self._batch_size), :]
                feed_dict = {self.tf_train_dataset: batch_data, self.tf_train_labels: batch_labels, self.tf_keep_prob: k_prob}

                _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)

#                if batch % 100 == 0:
#                    summary_str = session.run(merged)
#                    writer.add_summary(summary_str, batch)

                # Print out status of run
                if (batch % 500 == 0):
                    print("Minibatch loss at batch {}: {}".format(batch, l))

                    train_accuracy = self.accuracy(predictions, batch_labels)
                    print("Minibatch accuracy: {:.1f}".format(train_accuracy))

                    if self._use_valid:
                        valid_accuracy = self.accuracy(self.valid_prediction.eval(), self._ds.valid_labels)
                        print("Validation accuracy: {:.1f}".format(valid_accuracy))
                    else:
                        valid_accuracy = 0

                    self.epochs[batch] = EpochValue(batch, train_accuracy, valid_accuracy, l)

            # Evaluation the model against the test data set
            eval = self.test_prediction.eval()
            self.test_preds = eval.ravel()
            accuracy = self.accuracy(eval, self._ds.test_labels)
            self.params['accuracy'] = accuracy

            print("Test accuracy ({}): {:.1f}".format(self.name, accuracy))

