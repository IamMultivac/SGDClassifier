import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator

class SGD(ClassifierMixin, BaseEstimator):

    def __init__(self, learning_rate = .001, epochs = 1000, random_state = 42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def feedforward_backpropagation(self, X, y, W, b):
        """
        X_nxm
        W_mX1
        """

        m = X.shape[0]


        A = self.sigmoid(np.dot(W.T, X) + b)


        loss = -1/m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        dw = 1/m * np.dot(X, (A - y).T)
        db = 1/m* np.sum(A - y)

        

        return loss, dw, db


    def fit(self, X, y):

        self.cost_ = list()

        np.random.seed(self.random_state)

        self.W = np.random.random((X.shape[1],1))
        self.b = np.random.random()

        for i in range(self.epochs):
            # print(f"Epoch: {i}")
            loss, dw, db = self.feedforward_backpropagation(X.T, y, self.W, self.b)
            self.cost_.append(loss)
            self.W = self.W - dw * self.learning_rate
            self.b = self.b - db * self.learning_rate

    

        return self

    def predict_proba(self, X, y = None):

        negatives =  self.sigmoid(np.dot(self.W.T,X.T) + self.b).T
        positives = 1 - negatives

        return np.hstack([positives, negatives])

    def predict(self, X, y = None, threshold = .5):

        probas = self.sigmoid(np.dot(self.W.T,X.T) + self.b).T

        return np.where(probas >= threshold, 1, 0)
