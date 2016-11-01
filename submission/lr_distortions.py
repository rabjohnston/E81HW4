from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from six.moves import cPickle
from six.moves import range
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.linear_model
from sklearn.decomposition import PCA
import tensorflow as tf

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size * 3)).astype(np.float32)
    labels = labels    
    return dataset, labels

f1 = open('cifar-10-batches-py/data_batch_1', 'rb')
f2 = open('cifar-10-batches-py/data_batch_2', 'rb')
f3 = open('cifar-10-batches-py/data_batch_3', 'rb')
f4 = open('cifar-10-batches-py/data_batch_4', 'rb')
f5 = open('cifar-10-batches-py/data_batch_5', 'rb')
ftest = open('cifar-10-batches-py/test_batch', 'rb')

datadict1 = cPickle.load(f1,encoding='bytes')
datadict2 = cPickle.load(f2,encoding='bytes')
datadict3 = cPickle.load(f3,encoding='bytes')
datadict4 = cPickle.load(f4,encoding='bytes')
datadict5 = cPickle.load(f5,encoding='bytes')
datadictT = cPickle.load(ftest,encoding='bytes')

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
ftest.close()

X1 = datadict1[b"data"]  #b prefix is for bytes string literal.
Y1 = datadict1[b'labels']
X2 = datadict2[b"data"]  #b prefix is for bytes string literal.
Y2 = datadict2[b'labels']
X3 = datadict3[b"data"]  #b prefix is for bytes string literal.
Y3 = datadict3[b'labels']
X4 = datadict4[b"data"]  #b prefix is for bytes string literal.
Y4 = datadict4[b'labels']
X5 = datadict5[b"data"]  #b prefix is for bytes string literal.
Y5 = datadict5[b'labels']
Xtest = datadictT[b"data"]  #b prefix is for bytes string literal.
Ytest = datadictT[b'labels']

# convert pixel values to a numpy array of normalized floats  
unshaped_floats1 = np.array(X1, dtype=float) / 255.0 - 0.5
unshaped_floats2 = np.array(X2, dtype=float) / 255.0 - 0.5
unshaped_floats3 = np.array(X3, dtype=float) / 255.0 - 0.5
unshaped_floats4 = np.array(X4, dtype=float) / 255.0 - 0.5
unshaped_floats5 = np.array(X5, dtype=float) / 255.0 - 0.5
unshaped_floats = np.vstack((unshaped_floats1, unshaped_floats2, unshaped_floats3, unshaped_floats4, unshaped_floats5))
unshaped_floatsT = np.array(Xtest, dtype=float) / 255.0 - 0.5

train_dataset = unshaped_floats
test_dataset = unshaped_floatsT
train_labels = np.hstack((Y1, Y2, Y3, Y4, Y5))
test_labels = np.hstack((Ytest))

train_dataset, valid_dataset, train_labels, valid_labels = \
            train_test_split(train_dataset, train_labels, test_size=.25, random_state=10)

image_size = 32
num_labels = 10

# "unpacking" the image into format which could be use for image distortions
test_dataset = test_dataset.reshape([-1, 3, 32, 32])
test_dataset = test_dataset.transpose([0, 2, 3, 1])

# Use 1 of out the 3 different disstortions bellow for each of the runs 
test_dataset = np.array([np.rot90(img) for img in test_dataset])        
# data = np.array([np.fliplr(img) for img in data]) 
# data = np.array([np.flipud(img) for img in data]) 

# "packing" the images back to initial format
test_dataset = test_dataset.transpose([0, 3, 1, 2])
test_dataset, test_labels = reformat(test_dataset, test_labels)

X_train = train_dataset
y_train = train_labels
X_valid = valid_dataset
y_valid = valid_labels
X_test = test_dataset
y_test = test_labels
    
clf = LogisticRegression(C=0.01, n_jobs=-1).fit(X_train, y_train)
# For KNN classifier replace the line above with the following one:
# clf = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1).fit(X_train, y_train)   
print("Validation accuracy:", clf.score(X_valid, y_valid))
print("Test accuracy:", clf.score(X_test, y_test))  