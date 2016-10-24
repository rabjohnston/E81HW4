
import time
import pickle
import matplotlib.pyplot as plt
from dataset import DataSet
from baseline_nn import Baseline_nn
from baseline_cnn import Baseline_cnn

def plot():
    # Plot all ROC curves
    plt.figure()
    for i, clf in zip(range(len(classifiers)), classifiers):
        fpr, tpr, _ = roc_curve(test_labels.ravel(), test_preds[clf])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 label='ROC curve ' + clf + ' (area = {0:0.4f})'
                                            ''.format(roc_auc),
                 color=dark2_colors[i], linestyle='-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of multiclass micro-average ROC curves')
    plt.legend(loc="lower right", fontsize=14)
    plt.show()


def run(model, num_epochs=50000):

    start = time.time()

    model.run_session(num_epochs)
    pickle.dump(model, open( "{}.p".format(model.name), "wb" ))

    print("Completed in {} minutes".format(round((time.time( ) -start ) /60, 2)))


def main():
    flatDataSet = DataSet()
    flatDataSet.load()

    shapedDataSet = DataSet()
    shapedDataSet.load(False)

    # Baseline Neural Network
    nn = Baseline_nn(flatDataSet)
    nn.create()
    run(nn)

    # Baseline CNN
    cnn = Baseline_cnn(shapedDataSet)
    cnn.create()
    run(cnn)

    # Visualise the data for one image
    # d.display(40000)

    print('Finished')

if __name__ == '__main__':
    main()

# Default
# Test accuracy: 99.3
# Completed in 16.42 minutes
