import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class KMeans:
    def __init__(self, n_clusters=4, max_iters=1000000, tol=1e-20):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.cluster_labels = None

    def initialize_centroids(self, image_database):
        features = image_database.features
        labels = image_database.labels
        # Inicializar centroides usando los arrays de features y labels
        initial_centroids = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 0:
                initial_centroids.append(features[label_indices[0]])
            else:
                # Si no hay puntos en el cluster, inicializar aleatoriamente
                initial_centroids.append(features[np.random.choice(len(features))])
        self.centroids = np.array(initial_centroids)

    def fit(self, X):
        n_samples = X.shape[0]
        old_labels = np.zeros(n_samples)
        
        for _ in range(self.max_iters):
            distances = ((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2)
            new_labels = np.argmin(distances, axis=1)
            
            if np.sum(new_labels != old_labels) <= self.tol * n_samples:
                break
                
            old_labels = new_labels
            self.labels = new_labels
            
            for i in range(self.n_clusters):
                mask = self.labels == i
                if np.any(mask):
                    self.centroids[i] = np.mean(X[mask], axis=0)
        
        self.points = X
        return self
        

    def assign_cluster_labels(self, true_labels):
        self.cluster_labels = []
        for i in range(self.n_clusters):
            cluster_mask = self.labels == i
            if np.any(cluster_mask):
                cluster_true_labels = true_labels[cluster_mask]
                most_common = Counter(cluster_true_labels).most_common(1)[0]
                self.cluster_labels.append(most_common[0])
            else:
                self.cluster_labels.append(None)
        
        return self.cluster_labels

    def predict(self, X):
        X = X.reshape(1, -1)
        distances = np.sqrt(np.sum((self.centroids - X) ** 2, axis=1)).reshape(1, -1)
        return np.argmin(distances)
    

    def visualizar_pca(self, ax=None, new_point=None):
        # Reducir dimensionalidad con PCA
        pca = PCA(n_components=3)
        transformed_centroids = pca.fit_transform(self.centroids)
        transformed_points = pca.transform(self.points)

        # Usar el axis proporcionado o crear uno nuevo si no se proporciona
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Crear mapa de colores
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n_clusters))

        # Plotear puntos para cada cluster
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_points = transformed_points[mask]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                      label=f'Cluster {i}', color=colors[i])
            # Plotear centroide
            ax.scatter(transformed_centroids[i, 0], transformed_centroids[i, 1], transformed_centroids[i, 2], 
                      marker='^', s=200, color=colors[i])

        # Marcar el nuevo punto si está presente
        if new_point is not None:
            transformed_new_point = pca.transform(new_point.reshape(1, -1))
            ax.scatter(transformed_new_point[0, 0], transformed_new_point[0, 1], transformed_new_point[0, 2],
                       marker='x', color='red', s=200, label='Imagen Nueva')

        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        ax.legend()
        return ax