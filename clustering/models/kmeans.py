from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from clustering.path_definitions import PROCESSED_DATA_PATH


def fast_euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a - b) ** 2
    return diff.sum(axis=-1) ** 0.5


def distance_matrix(XX: np.ndarray, YY: np.ndarray):
    distances = []
    for x in XX:
        for y in YY:
            distances.append(fast_euclidean_dist(x, y))
    return np.array(distances).reshape(XX.shape[0], -1)


def kmeans(n_clusters: int, X: np.ndarray):

    # Init
    n_samples, n_features = X.shape

    # Initial random centroids
    ii = np.random.choice(n_samples, size=n_clusters)
    random_centroids = X[ii]

    # Obtain Labels

    distances = distance_matrix(X, random_centroids)
    labels = [np.argmin(sample) for sample in distances]




    pass

    #     # New centroids
    #     new_centers = np.array(
    #         [X[np.asarray(labels) == i].mean(0) for i in range(n_clusters)])
    #
    #     # Convergence
    #     if np.all(centers == new_centers):
    #         break
    #     centers = new_centers
    #
    # return centers, labels


p = PROCESSED_DATA_PATH / 'vowel.csv'
vowel_df = pd.read_csv(p,  index_col=0)

kmeans(11, vowel_df.values)
kk = KMeans(11)
kk.fit(vowel_df)
print(kk)
