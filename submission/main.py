
import time
import sys
import traceback
import datetime
import numpy as np
import pickle
from optimizer_params import *
import tensorflow as tf

#import matplotlib.pyplot as plt
from dataset import DataSet
from splitdataset import SplitDataset2
from baseline_nn import Baseline_nn
from baseline_cnn import Baseline_cnn
from baseline_axn import Baseline_axn
from baseline_axn2 import Baseline_axn2
from baseline_lr import Baseline_lr

def run(model, iteration, num_batches=50000):

    try:
        start = time.time()

        model.run_session(num_batches)

        timeCompleted = round((time.time( ) -start ) /60, 2)
        print("Completed in {} minutes".format(timeCompleted))
        model.params['time'] = timeCompleted

        # Save
        base_filename = 'saved_{}_{:03d}'.format(model.name, iteration)

        print(base_filename)

        np.save('{}.preds'.format(base_filename), model.test_preds)
        pickle.dump(model.params, open('{}.params'.format(base_filename), 'wb'))
        pickle.dump(model.optimizer_params, open('{}.opt'.format(base_filename), 'wb'))
        pickle.dump(model.epochs, open('{}.epochs'.format(base_filename), 'wb'))
    except:
        e = sys.exc_info()[0]
        print('Exception encountered: ', e)
        print(traceback.format_exc())


def searchForNNParams(ds, num_epochs=30000):

    # Define Hyper-parameters
    batch_sizes = { 64, 128, 256 }
    hidden_nodes = {512, 1024, 2048, 4096}
    lamb_regs = {0.005, 0.01, 0.05}

    iteration = 0

    # Perform a Grid Search on all hyper-parameters
    for batch_size in batch_sizes:
        for hidden_node in hidden_nodes:
            for lamb_reg in lamb_regs:
                nn = Baseline_nn(ds)
                nn.create(batch_size=batch_size, hidden_nodes=hidden_node, lamb_reg=lamb_reg)
                run(nn, iteration, num_epochs)
                nn = None
                iteration += 1


def searchForCNNParams(ds, num_batches=30000):
    # Define Hyper-parameters
    batch_sizes = { 8, 16, 32, 64, 128, 256 }
    patch_sizes = { 5, 6, 8, 10, 12 }

    # This seriously chews up memory
    depth1s = { 32, 64 }
    depth2s = { 64, 128 }
    num_hiddens = { 1024, 2048 }

    iteration = 0

    # Perform a Grid Search on all hyper-parameters
    for batch_size in batch_sizes:
        for patch_size in patch_sizes:
            for depth1 in depth1s:
                for depth2 in depth2s:
                    for num_hidden in num_hiddens:
                        cnn = Baseline_cnn(ds)
                        cnn.create(batch_size=batch_size,
                                   patch_size=patch_size,
                                   depth1=depth1,
                                   depth2=depth2,
                                   num_hidden=num_hidden)
                        run(cnn, iteration, num_batches)
                        cnn = None
                        iteration += 1


def searchForAXN2AdamParams(ds, num_batches=50000):
    # Define Hyper-parameters

    learning_rates = { 0.1, 0.01, 0.001, 0.0001 }
    beta1s = { 0.8, 0.9, 0.99, 0.999 }
    beta2s = { 0.8, 0.9, 0.99, 0.999 }

    epsilons = { 1e-10, 1e-8, 1e-6, 1e-4 }


    iteration = 200

    # Perform a Grid Search on all hyper-parameters
    for learning_rate in learning_rates:
        for beta1 in beta1s:
            for beta2 in beta2s:
                for epsilon in epsilons:
                    cnn = Baseline_axn2(ds)
                    cnn.create(optimizer_params=AdamParams(learning_rate=learning_rate,
                                          beta1=beta1,
                                          beta2=beta2,
                                          epsilon=epsilon))
                    run(cnn, iteration, num_batches)
                    iteration += 1


def searchForAXN2AdadeltaParams(ds, num_batches=50000):
    # Define Hyper-parameters

    learning_rates = { 0.1, 0.01, 0.001, 0.0001 }
    rhos = { 0.8, 0.9, 0.95, 0.999 }

    epsilons = { 1e-10, 1e-8, 1e-6, 1e-4 }


    iteration = 300

    # Perform a Grid Search on all hyper-parameters
    for learning_rate in learning_rates:
        for rho in rhos:
            for epsilon in epsilons:
                cnn = Baseline_axn2(ds)
                cnn.create(optimizer_params=AdadeltaParams(learning_rate=learning_rate,
                                      rho=rho,
                                      epsilon=epsilon))
                run(cnn, iteration, num_batches)
                iteration += 1


def searchForAXN2GDParams(ds, num_batches=50000):
    # Define Hyper-parameters

    learning_rates = { 0.1, 0.01, 0.001, 0.0001 }

    iteration = 0

    # Perform a Grid Search on all hyper-parameters
    for learning_rate in learning_rates:
        axn2 = Baseline_axn2(ds)
        axn2.create(optimizer_params=GradientDescentParams(learning_rate=learning_rate))
        run(axn2, iteration, num_batches)
        iteration += 1


def searchForAXN2AdagradParams(ds, num_batches=50000):
    # Define Hyper-parameters

    learning_rates = { 0.1, 0.01, 0.001, 0.0001 }
    initial_accumulator_values = {0.001, 0.01, 0.1, 1}

    iteration = 100

    # Perform a Grid Search on all hyper-parameters
    for learning_rate in learning_rates:
        for initial_accumulator_value in initial_accumulator_values:
            axn2 = Baseline_axn2(ds)
            axn2.create(optimizer_params=AdagradParams(learning_rate=learning_rate,
                                                       initial_accumulator_value=initial_accumulator_value))
            run(axn2, iteration, num_batches)
            iteration += 1


def runNNs():
    flatDataSet = DataSet()
    flatDataSet.load()

    # Baseline Neural Network
    nn = Baseline_nn(flatDataSet)
    nn.create(start_learning_rate = 0.01)
    run(nn, 0, num_batches=50000)

    # Search for parameters
    #searchForNNParams(flatDataSet)


def runCNNs():

    shapedDataSet = DataSet()
    shapedDataSet.load(False)

    # Baseline CNN
    cnn = Baseline_cnn(shapedDataSet)
    cnn.create()
    run(cnn, 0, num_batches=50000)

    # Search for parameters
    #searchForCNNParams(shapedDataSet, num_batches=60000)


def runAXNs():

    shapedDataSet = DataSet(use_valid=False)
    shapedDataSet.load(False)

    # Baseline AXN
    axn = Baseline_axn(shapedDataSet)
    axn.create(AdamParams(), batch_size=128)
    run(axn, 0, num_batches=50000)

    # Search for parameters
    #searchForAXNParams(shapedDataSet, num_batches=1000)


def runSplitNNs():

    def runSingleNN(ds, iteration):
        axn = Baseline_nn(ds)
        axn.create()
        run(axn, iteration, num_batches=20000)

    ds = DataSet(use_valid=False)
    ds.load()

    split = SplitDataset()
    split.load(ds.train_dataset, ds.train_labels)

    ds.train_dataset = split.a_dataset
    ds.train_labels = split.a_labels
    runSingleNN(ds,0)

    ds.train_dataset = split.ab_dataset
    ds.train_labels = split.ab_labels
    runSingleNN(ds, 1)

    ds.train_dataset = split.abc_dataset
    ds.train_labels = split.abc_labels
    runSingleNN(ds, 2)

    ds.train_dataset = split.abcd_dataset
    ds.train_labels = split.abcd_labels
    runSingleNN(ds, 3)


def runSplitCNNs():

    def runSingleCNN(ds, iteration):
        axn = Baseline_cnn(ds)
        axn.create()
        run(axn, iteration, num_batches=20000)

    ds = DataSet(use_valid=False)
    ds.load(False)

    split = SplitDataset()
    split.load(ds.train_dataset, ds.train_labels)

    ds.train_dataset = split.a_dataset
    ds.train_labels = split.a_labels
    runSingleCNN(ds,0)

    ds.train_dataset = split.ab_dataset
    ds.train_labels = split.ab_labels
    runSingleCNN(ds, 1)

    ds.train_dataset = split.abc_dataset
    ds.train_labels = split.abc_labels
    runSingleCNN(ds, 2)

    ds.train_dataset = split.abcd_dataset
    ds.train_labels = split.abcd_labels
    runSingleCNN(ds, 3)


def runSplitAXNs():

    def runSingleAXN(ds, iteration):
        axn = Baseline_axn(ds)
        axn.create(AdamParams())
        run(axn, iteration, num_batches=20000)

    shapedDataSet = DataSet(use_valid=False)
    shapedDataSet.load(False)

    split = SplitDataset()
    split.load(shapedDataSet.train_dataset, shapedDataSet.train_labels)

    shapedDataSet.train_dataset = split.a_dataset
    shapedDataSet.train_labels = split.a_labels
    runSingleAXN(shapedDataSet,0)

    shapedDataSet.train_dataset = split.ab_dataset
    shapedDataSet.train_labels = split.ab_labels
    runSingleAXN(shapedDataSet, 1)

    shapedDataSet.train_dataset = split.abc_dataset
    shapedDataSet.train_labels = split.abc_labels
    runSingleAXN(shapedDataSet, 2)

    shapedDataSet.train_dataset = split.abcd_dataset
    shapedDataSet.train_labels = split.abcd_labels
    runSingleAXN(shapedDataSet, 3)


def runAXN2s():

    shapedDataSet = DataSet(use_valid=False)
    shapedDataSet.load(False )

    # Baseline CNN
    #axn2 = Baseline_axn2(shapedDataSet)
    #axn2.create(GradientDescentParams())
    #run(axn2, 0, num_batches=50000)

    # Search for parameters
    #searchForAXNParams(shapedDataSet, num_batches=1000)

    # GD tends to increase in error over 20000 batches
    searchForAXN2GDParams(shapedDataSet, num_batches=20000)
    searchForAXN2AdadeltaParams(shapedDataSet, num_batches=20000)
    searchForAXN2AdagradParams(shapedDataSet, num_batches=20000)
    searchForAXN2AdamParams(shapedDataSet, num_batches=20000)


    # Baseline CNN
    #axn = Baseline_axn2(shapedDataSet)
    #axn.create(AdamParams(),batch_size=128)
    #run(axn, 0, num_batches=50000)

    # axn = Baseline_axn2(shapedDataSet)
    # axn.create(AdagradParams())
    # run(axn, 1, num_batches=50000)
    #
    # axn = Baseline_axn2(shapedDataSet)
    # axn.create(GradientDescentParams())
    # run(axn, 2, num_batches=50000)
    #
    # axn = Baseline_axn2(shapedDataSet)
    # axn.create(AdadeltaParams())
    # run(axn, 3, num_batches=50000)


def runSingleLR(lr, X_train, y_train, X_valid, y_valid, X_test, y_test, iteration = 0):
    start = time.time()

    lr.create(X_train, y_train, X_valid, y_valid, X_test, y_test)

    timeCompleted = round((time.time() - start) / 60, 2)
    print("Completed in {} minutes".format(timeCompleted))
    lr.params['time'] = timeCompleted

    base_filename = 'saved_lr_{:03d}'.format(iteration)
    np.save('{}.preds'.format(base_filename), lr.test_preds)
    pickle.dump(lr.params, open('{}.params'.format(base_filename), 'wb'))


def runLR():
    lr = Baseline_lr()
    X_train, y_train, X_valid, y_valid, X_test, y_test = lr.load()
    runSingleLR(lr, X_train, y_train, X_valid, y_valid, X_test, y_test)


def runSplitLR():
    lr = Baseline_lr()
    X_train, y_train, X_valid, y_valid, X_test, y_test = lr.load()

    spl = SplitDataset2()
    spl.load(X_train, y_train)

    print('LR on dataset A')
    runSingleLR(lr, spl.a_dataset, spl.a_labels, X_valid, y_valid, X_test, y_test, iteration = 1)

    print('LR on dataset AB')
    runSingleLR(lr, spl.ab_dataset, spl.ab_labels, X_valid, y_valid, X_test, y_test, iteration=2)

    print('LR on dataset ABC')
    runSingleLR(lr, spl.abc_dataset, spl.abc_labels, X_valid, y_valid, X_test, y_test, iteration=3)

    print('LR on dataset ABCD')
    runSingleLR(lr, spl.abcd_dataset, spl.abcd_labels, X_valid, y_valid, X_test, y_test, iteration=4)

def runBigAXN2():
    shapedDataSet = DataSet(use_valid=False)
    shapedDataSet.load(False )

    axn2 = Baseline_axn2(shapedDataSet)

    # Keep batch_size at 128 - it's half the time of 256
    axn2.create(optimizer_params=AdagradParams(learning_rate=0.001,
                                               initial_accumulator_value=0.01))
    # Run this half a million times
    run(axn2, 991, 5000)


def main():
    # Visualise the data for one image
    # d.display(40000)

    #runNNs()

    #
    #runCNNs()

    #runAXNs()

    #runAXN2s()

    #runSplitAXNs()
    #runSplitCNNs()

    #runLR()

    #runSplitLR()

    runBigAXN2()

    print('Finished')

if __name__ == '__main__':
    main()


