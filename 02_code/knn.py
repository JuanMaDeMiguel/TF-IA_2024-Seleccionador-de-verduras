import numpy as np
from collections import Counter

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Almacena los datos de entrenamiento"""
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        # Reshape X to ensure it's a row vector
        X = X.reshape(1, -1)

        # Calculate distances between X and all training points
        distances = np.sqrt(np.sum((self.X_train - X) ** 2, axis=1)).reshape(1, -1)
        
        # Get indices of k nearest neighbors for all points at once
        k_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        
        # Get the labels of k nearest neighbors for all points
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote for each point
        predictions = np.array([Counter(labels).most_common(1)[0][0] 
                              for labels in k_nearest_labels])
        
        return predictions
