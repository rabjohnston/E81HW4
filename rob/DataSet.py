import numpy as np
from six.moves import cPickle as pickle


class DataSet():

    def __init__(self):
        self.label_name = None
        self.data = None
        self.labels = None
        self.image = None

        self.valid_dataset
        self.train_dataset


    def unpickle(self, file):
        fo = open(file, 'rb')
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
        return dict

    def to_image(self, X):
        return X.reshape((X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    def load(self):
        b1 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b2 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b3 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b4 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b5 = self.unpickle('cifar-10-batches-py/data_batch_1')

        self.data = np.vstack([b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data']])
        self.image = self.to_image(self.data)
        print(self.data.shape)

        self.labels = np.hstack([b1[b'labels'], b2[b'labels'], b3[b'labels'], b4[b'labels'], b5[b'labels']])

        meta = self.unpickle('cifar-10-batches-py/batches.meta')

        self.label_name = meta[b'label_names']
        print(self.labels.shape)
        #print(test[b'label_names'])


    def display(self, id):
        print('Display')
        import matplotlib.pyplot as plt
        plt.imshow(self.image[id])
        plt.title(self.label_name[self.labels[id]])
        plt.show()