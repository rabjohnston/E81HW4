
import time
import pickle
#import matplotlib.pyplot as plt
from dataset import DataSet
from baseline_nn import Baseline_nn
from baseline_cnn import Baseline_cnn

# def plot():
#     # Plot all ROC curves
#     plt.figure()
#     for i, clf in zip(range(len(classifiers)), classifiers):
#         fpr, tpr, _ = roc_curve(test_labels.ravel(), test_preds[clf])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr,
#                  label='ROC curve ' + clf + ' (area = {0:0.4f})'
#                                             ''.format(roc_auc),
#                  color=dark2_colors[i], linestyle='-', linewidth=2)
#
#     plt.plot([0, 1], [0, 1], 'k--', lw=2)
#     plt.xlim([-0.1, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Comparison of multiclass micro-average ROC curves')
#     plt.legend(loc="lower right", fontsize=14)
#     plt.show()


def run(model, num_epochs=50000):

    start = time.time()

    model.run_session(num_epochs)
    #pickle.dump(model, open( "{}.p".format(model.name), "wb" ))

    print("Completed in {} minutes".format(round((time.time( ) -start ) /60, 2)))

def searchForNNParams(ds, num_epochs=50000):
    batch_sizes = { 64, 128, 256 }
    hidden_nodes = {512, 1024, 2048, 4096}
    lamb_regs = {0.005, 0.01, 0.05}

    for batch_size in batch_sizes:
        for hidden_node in hidden_nodes:
            for lamb_reg in lamb_regs:
                nn = Baseline_nn(ds)
                nn.create(batch_size=batch_size, hidden_nodes=hidden_node, lamb_reg=lamb_reg)
                run(nn, num_epochs)

def searchForCNNParams(ds, num_epochs=50000):

    batch_sizes = { 8, 16, 32, 64 }
    patch_sizes = { 5, 6, 8, 10, 12 }
    depth1s = { 32, 64, 128 }
    depth2s = { 64, 128, 256 }
    num_hiddens = { 1024, 2048, 3072, 4096 }

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
                        run(cnn, num_epochs)


def main():
    flatDataSet = DataSet()
    flatDataSet.load()

    shapedDataSet = DataSet()
    shapedDataSet.load(False)

    # Baseline Neural Network
    #nn = Baseline_nn(flatDataSet)
    #nn.create()
    #run(nn)

    # Search for parameters
    searchForNNParams(flatDataSet, 1)

    # Baseline CNN
    #cnn = Baseline_cnn(shapedDataSet)
    #cnn.create()
    #run(cnn)

    # Search for parameters
    searchForCNNParams(shapedDataSet, 1)


    # Visualise the data for one image
    # d.display(40000)

    print('Finished')

if __name__ == '__main__':
    main()

# Default
# Test accuracy: 99.3
# Completed in 16.42 minutes
