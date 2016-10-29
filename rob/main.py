
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
from baseline_nn import Baseline_nn
from baseline_cnn import Baseline_cnn
from baseline_axn import Baseline_axn
from baseline_axn2 import Baseline_axn2


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
    batch_sizes = { 64, 128, 256, 128, 256 }
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
    #nn = Baseline_nn(flatDataSet)
    #nn.create()
    #run(nn, 0, num_epochs=50000)

    # Search for parameters
    searchForNNParams(flatDataSet)


def runCNNs():

    shapedDataSet = DataSet()
    shapedDataSet.load(False)

    # Baseline CNN
    # cnn = Baseline_cnn(shapedDataSet)
    # cnn.create(batch_size=8, patch_size=)
    # run(cnn, 0, num_epochs=30000)

    # Search for parameters
    searchForCNNParams(shapedDataSet, num_batches=60000)


def runAXNs():

    shapedDataSet = DataSet()
    shapedDataSet.load(False)

    # Baseline CNN
    axn = Baseline_axn(shapedDataSet)
    axn.create(AdamParams())
    run(axn, 0, num_batches=3)

    # Search for parameters
    #searchForAXNParams(shapedDataSet, num_batches=1000)


def runAXN2s():

    shapedDataSet = DataSet(use_valid=True)
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



def main():
    # Visualise the data for one image
    # d.display(40000)

    #runNNs()

    #
    #runCNNs()

    #runAXNs()

    runAXN2s()



    print('Finished')

if __name__ == '__main__':
    main()


