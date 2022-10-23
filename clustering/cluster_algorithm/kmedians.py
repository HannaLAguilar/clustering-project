import numpy as np
import pandas as pd
<<<<<<< HEAD
import itertools

import sys
sys.path.append('C:/Users/mario/Github/UPC/IML - Introduction to Machine Learning/Work 1 - Clustering Excercise/')

from clustering.path_definitions import PROCESSED_DATA_PATH
from clustering.get_data.preprocessing import import_data
from time import time
from typing import Dict, List, Tuple

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Internal metrics
INTERNAL_METRICS = {'calinski': metrics.calinski_harabasz_score,
                    'davies': metrics.davies_bouldin_score,
                    'silhouette': metrics.silhouette_score}

# External metrics
EXTERNAL_METRICS = {'ARI': metrics.adjusted_rand_score,
                    'AMI': metrics.adjusted_mutual_info_score,
                    'homo': metrics.homogeneity_score,
                    'compl': metrics.completeness_score,
                    'v-measure': metrics.v_measure_score}

def k_medians(X, k, n_iter, epsilon, limit, verbose = False):
    centers = X[np.random.randint(0, len(X), k)]
    iter = 0
    no_change = 0
    past_centers = np.ones_like(centers)
    while n_iter != -1:
        iter += 1

        distances = np.array([[sum(abs(instance - center)) for instance in X] for center in centers])
        close_cluster = np.argmin(distances, axis=0)

        for i in range(k):
            cluster = X[close_cluster == i]
            if cluster.shape[0] == 0:
                centers[i] = X[np.random.randint(0, len(X), 1)]
            else:
                centers[i] = np.median(cluster, axis=0)

        if (abs(np.sum(past_centers - centers))) < epsilon:
            no_change += 1
        else:
            no_change = 0
        past_centers = centers.copy()
        if iter == n_iter:
            if verbose:
                print('Max numer of iterations reached! Last calculated cluster centers returned.')
            n_iter = -1
        elif no_change == limit:
            if verbose:
                print('Limit of iterations without significant improvement reached at iter {}. Convergence got it!'.format(iter))
            n_iter = -1
    return centers, close_cluster

def hyperparameter_clustering(X: np.ndarray,
                              y_true_num: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              EXTERNAL_METRICS.values()) -> Tuple[List, List, float]:

    t0 = time()
    da = []
    params_combs = list(itertools.product(*list(parameters.values())))
    param = {}
    KMD_clustering = []
    for param_comb in params_combs:
        for index, p in enumerate(list(parameters.keys())):
            param[p] = param_comb[index]

        t0 = time()
        # Perform clustering
        centers, labels = k_medians(X.values,param['n_clusters'],param['n_iter'], param['epsilon'], param['limit'], param['verbose'])
        tf = time() - t0

        KMD_clustering.append(labels)
        # Save in a list
        result = [tf, *param_comb]
        # Internal index metrics
        result += [m(X, labels)
                   for m in internal_metrics]
        # External index metrics
        result += [m(y_true_num, labels)
                   for m in external_metrics]
        da.append(result)

    func_time = time() - t0
    return da, KMD_clustering, func_time


def main(data_name: str, save: bool = True):
    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)

    X = df.iloc[:, :-1]

    y_true = df['y_true']
    le = LabelEncoder().fit(y_true.values)
    y_true_num = le.transform(y_true)

    # Parameters for cluster
    params = {'n_clusters': [9,10,11,12,13],
              'n_iter': [100],
              'epsilon': [0.000001],
              'limit':[10],
              'verbose':['True']}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true_num, params)
    metric_data, clus, global_time = da
    columns = ['time', *list(params.keys())]
    columns = columns + list(INTERNAL_METRICS.keys()) + list(
        EXTERNAL_METRICS.keys())

    # # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'kmedians_results_{data_name}')

    return metric_df, global_time


if __name__ == '__main__':
    DATASET_NAME = 'vowel.csv'
    METRICS_DF, GLOBAL_TIME = main(DATASET_NAME)
    print(METRICS_DF)
    print(GLOBAL_TIME)



=======

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
>>>>>>> 94dc0fe910621ba6f268c4b22429d36f69df53aa
