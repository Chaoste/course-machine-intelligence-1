import numpy as np
import scipy.stats

from .algorithm import Algorithm

class KNN(Algorithm):

    def pca(self, test_data):
        # Merge training and test set to calculate their common PCs
        data = np.concatenate([self.train_data, test_data])
        # Reduce DATA using its K principal components
        data = data.astype('float64')
        data -= np.mean(data, axis=0)
        U, S, V = np.linalg.svd(data, full_matrices=False)
        data = U[:, :self.k_pca].dot(np.diag(S)[:self.k_pca, :self.k_pca])
        # Again split up dataset into training and test set
        train_data = data[:len(self.train_data)]
        test_data = data[len(self.train_data):]
        return train_data, test_data

    def __init__(self, k=6, k_pca=45):
        self.train_data = None
        self.train_labels = None
        self.k = k
        self.k_pca = k_pca

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def eval(self, test_data, pca=True):
        if pca:
            train_data, test_data = self.pca(test_data)
        predictions = []
        for sample in test_data:
            differences = (train_data - sample)
            distances = np.einsum('ij, ij->i', differences, differences)
            nearest = self.train_labels[np.argsort(distances)[:self.k]]
            predictions.append(scipy.stats.mode(nearest)[0][0])
        return predictions
