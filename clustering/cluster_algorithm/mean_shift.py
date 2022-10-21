import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from clustering.path_definitions import PROCESSED_DATA_PATH
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


def hyperparameter_clustering(X: np.ndarray,
                              y_true_num: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              EXTERNAL_METRICS.values()) -> \
        Tuple[List, List, float]:

    t0 = time()
    agglo_clustering = []
    da = []
    params_combs = list(itertools.product(*list(parameters.values())))
    # ind_params = {}
    for param_comb in params_combs:
        # for index, param in enumerate(list(parameters.keys())):
        #     ind_params[param] = param_comb[index]
        t0 = time()

        # Perform clustering
        bandwidth = estimate_bandwidth(X, quantile=param_comb[0], n_samples=param_comb[1], random_state=param_comb[2], n_jobs=param_comb[3])

        ms = MeanShift(bin_seeding=param_comb[4], n_jobs=param_comb[3])
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        tf = time() - t0
        agglo_clustering.append(clustering)
        # Save in a list
        result = [tf, ms.labels_, ms.cluster_centers_, k]
        # Internal index metrics
        result += [m(X, clustering.labels_)
                   for m in internal_metrics]
        # External index metrics
        result += [m(y_true_num, clustering.labels_)
                   for m in external_metrics]
        da.append(result)

    func_time = time() - t0
    return da, agglo_clustering, func_time


def main(data_name: str, n_clusters, save: bool = True):
    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1]
    y_true = df['y_true']
    le = LabelEncoder().fit(y_true.values)
    y_true_num = le.transform(y_true)

    # Parameters for cluster
    params = {'affinity': ['euclidean', 'cosine'],
              'linkage': ['single', 'complete', 'average'],
              'n_clusters': n_clusters}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true_num, params)
    metric_data, clus, global_time = da
    columns = ['time', 'affinity', 'linkage', 'n_clusters']
    columns = columns + list(INTERNAL_METRICS.keys()) + list(
        EXTERNAL_METRICS.keys())

    # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'agglo_results_{data_name}')

    return metric_df, global_time


if __name__ == '__main__':
    DATASET_NAME = 'pen-based.csv'
    N_CLUSTERS = list(range(5, 16, 1))
    METRICS_DF, GLOBAL_TIME = main(DATASET_NAME, N_CLUSTERS)