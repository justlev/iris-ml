import random
from scipy.spatial import distance

class ScrappyKNN():
    def euc(self,a,b):
        return distance.euclidean(a,b)

    def fit(self, x,y):
        self.X_train = x
        self.Y_train = y

    def predict(self, x):
        predictions = []
        for row in x:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = self.euc(row, self.X_train[0])
        best_index=0
        for i in range(1, len(self.X_train)):
            dist = self.euc(row, self.X_train[i])
            if dist<best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]
        