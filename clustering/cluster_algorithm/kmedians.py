import numpy as np
import pandas as pd

from clustering.cluster_algorithm.kmeans import (distance_matrix,
                                                 get_inertia, Results)


def run_kmedians(n_clusters: int,
                 X: np.ndarray,
                 init_centers: np.ndarray,
                 max_iteration,
                 verbose=False):
    # Data
    n_samples, n_features = X.shape
    df = pd.DataFrame(X)

    # Init. parameters
    centers = None
    labels = None
    inertia = None
    new_centroids = np.zeros((n_clusters, n_features))
    ii = 0

    while ii < max_iteration:
        if ii == 0:
            centers = init_centers
        else:
            centers = new_centroids

        # Distances
        distances = distance_matrix(X, centers)

        # Obtain Labels
        labels = np.array([np.argmin(sample) for sample in distances])

        df['labels'] = labels
        new_centroids = df.groupby('labels').median().values

        # Get inertia
        inertia = get_inertia(distances, labels, n_samples)

        if verbose:
            print(f'Iteration {ii}, Inertia: {inertia}')

        # Convergence
        if np.all(new_centroids == centers):
            centers = new_centroids
            if verbose:
                print(f'Converged iteration {ii + 1}, Inertia: {inertia}')
            break
        ii += 1

    return centers, labels, inertia


def kmedians(n_clusters: int,
             X: np.ndarray,
             max_iteration=300,
             n_attempt=10,
             verbose=False):
    # Data
    n_samples, n_features = X.shape

    # Init parameters
    best_inertia = None
    results = None
    best_attempt = None

    for i in range(n_attempt):
        if verbose:
            print(f'-----Attempt {i + 1}-----')

        # Initial random centroids from samples
        ii = np.random.choice(n_samples, size=n_clusters)
        random_centroids = X[ii]

        # Run kmeans
        centers, labels, inertia = run_kmedians(n_clusters, X,
                                                random_centroids,
                                                max_iteration,
                                                verbose)

        # Select best inertia
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_attempt = i
            results = Results(labels=labels,
                              centers=centers,
                              inertia=inertia)
    if verbose:
        print(f'Best attempt {best_attempt + 1}')

    return results
