import numpy as np
import pandas as pd
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



