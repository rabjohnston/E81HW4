
import time

from dataset import DataSet
from baseline_nn import Baseline_nn
from baseline_cnn import Baseline_cnn

def main():
    d = DataSet()
    d.load(False)

    b = Baseline_cnn(d)
    b.create()

    start = time.time()
    b.run_session(100000, "NN")
    print("Completed in {} minutes".format(round((time.time( ) -start ) /60, 2)))
    # Visualise the data for one image
    # d.display(40000)

    print('Finished')

if __name__ == '__main__':
    main()

# Default
# Test accuracy: 99.3
# Completed in 16.42 minutes
