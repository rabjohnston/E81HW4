import Results


def main():
    """
    Unit test code for Results class
    :return:
    """
    dir = 'results/NN20161025'

    # Load in the results
    rs = Results()
    rs.load(dir)

    # Search for all results that had a batch_size of 256
    search1 = {'batch_size': 256}
    find = rs.get_results(search1)
    for f in find:
        print('Result 1: ', f.params)

    # Search for all results that had a batch_size of 256, hidden_nodes of 2048 and lamb_reg of 0.005
    search2 = {'batch_size': 256, 'hidden_nodes': 2048, 'lamb_reg': 0.005}
    find = rs.get_results(search2)
    for f in find:
        print('Result 2: ', f.params)

    # For printing the ROC curves we get a dictionary of predictions for where the hidden_nodes rate is the only
    # unconstrained hyper-parameter
    # Search for all results that had a batch_size of 256, hidden_nodes of 2048 and lamb_reg of 0.005
    search3 = {'batch_size': 256, 'lamb_reg': 0.005}
    find = rs.get_results(search3)
    for f in find:
        print('Result 3: ', f.params)

    tp = rs.get_test_pred(find, 'hidden_nodes')

    for item in tp:
        print('Key: ', item)


    # Find the best accuracy
    print('Highest accuracy: ', rs.highest('accuracy').params)


if __name__ == '__main__':
    main()