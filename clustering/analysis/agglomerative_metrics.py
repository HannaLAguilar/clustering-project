from time import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from clustering.analysis import definitions
from clustering.path_definitions import PROCESSED_DATA_PATH


def hyperparameter_clustering(X: np.ndarray,
                              y_true: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              definitions.INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              definitions.EXTERNAL_METRICS.values()) -> \
        Tuple[List, List, float]:
    t0 = time()
    agglo_clustering = []
    da = []
    for affinity in parameters['affinity']:
        for linkage in parameters['linkage']:
            for k in parameters['n_clusters']:
                t0 = time()
                # Perform clustering
                clustering = AgglomerativeClustering(n_clusters=k,
                                                     affinity=affinity,
                                                     linkage=linkage).fit(X)
                tf = time() - t0
                agglo_clustering.append(clustering)
                # Save in a list
                result = [tf, affinity, linkage, k]
                # Internal index metrics
                result += [m(X, clustering.labels_)
                           for m in internal_metrics]
                # External index metrics
                result += [m(y_true, clustering.labels_)
                           for m in external_metrics]
                da.append(result)

    func_time = time() - t0
    return da, agglo_clustering, func_time


def get_metric_dataset(data_name: str,
                       n_clusters: int,
                       save: bool = True):
    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1].values
    y_true = df['y_true']

    # Parameters for cluster
    params = {'affinity': ['euclidean', 'cosine'],
              'linkage': ['single', 'complete', 'average'],
              'n_clusters': n_clusters}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true, params)
    metric_data, clus, global_time = da
    columns = ['time', 'affinity', 'linkage', 'n_clusters']
    columns = columns + list(definitions.INTERNAL_METRICS.keys()) + list(
        definitions.EXTERNAL_METRICS.keys())

    # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'agglo_results_{data_name}')

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
        get_metric_dataset(dataset_name, n_clusters)
