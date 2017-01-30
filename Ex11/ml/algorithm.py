import numpy as np

class Algorithm:
    def __init__():
        pass

    def fit(self, train_data, train_labels):
        raise NotImplementedError

    def eval(self, test_data):
        raise NotImplementedError

    def calc_accuracy(self, test_data, test_labels):
        predicted_labels = np.array(self.eval(test_data))
        return np.mean(predicted_labels == test_labels)
