from time import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from clustering.analysis import definitions
from clustering.path_definitions import PROCESSED_DATA_PATH
from clustering.cluster_algorithm.kmeans import kmeans


def hyperparameter_clustering(X: np.ndarray,
                              y_true_num: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              definitions.INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              definitions.EXTERNAL_METRICS.values()) -> \
        Tuple[List, float]:
    t_init = time()
    da = []

    for k in parameters['n_clusters']:
        # Perform clustering
        t0 = time()
        kmeans_result = kmeans(n_clusters=k, X=X)
        tf = time() - t0

        # Save in a list
        result = [k, tf, kmeans_result.inertia]

        # Internal index metrics
        result += [m(X, kmeans_result.labels)
                   for m in internal_metrics]

        # External index metrics
        result += [m(y_true_num, kmeans_result.labels)
                   for m in external_metrics]
        da.append(result)

    func_time = time() - t_init
    return da, func_time


def main(data_name: str, n_clusters, save: bool = True):

    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1].values
    y_true = df['y_true']
    le = LabelEncoder().fit(y_true.values)
    y_true_num = le.transform(y_true)

    # Parameters for cluster
    params = {'n_clusters': n_clusters}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true_num, params)
    metric_data, global_time = da
    columns = ['n_cluster', 'time', 'inertia']
    columns = columns + list(definitions.INTERNAL_METRICS.keys()) + list(
        definitions.EXTERNAL_METRICS.keys())

    # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'kmeans_results_{data_name}')

    return metric_df, global_time


if __name__ == '__main__':

    DATASET_NAME = ['iris.csv',
                    'vowel.csv',
                    'cmc.csv',
                    'pen-based.csv']

    N_CLUSTERS_RANGE = [definitions.n_clusters_iris,
                        definitions.n_clusters_vowel,
                        definitions.n_clusters_cmc,
                        definitions.n_clusters_pen]

    for dataset_name, n_clusters in zip(DATASET_NAME,
                                        N_CLUSTERS_RANGE):
        print(f'Processing metrics for {dataset_name} dataset')
        main(dataset_name, n_clusters)
