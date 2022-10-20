from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from time import time

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from clustering.path_definitions import PROCESSED_DATA_PATH
from clustering.visualization import visualize

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
                              internal_metrics: Dict = INTERNAL_METRICS.values(),
                              external_metrics: Dict = EXTERNAL_METRICS.values()):
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
                result += [m(y_true_num, clustering.labels_)
                           for m in external_metrics]
                da.append(result)

    func_time = time() - t0
    return da, agglo_clustering, func_time


def main(data_name: str, n_clusters, save: bool = True):
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1]
    y_true = df['y_true']
    le = LabelEncoder().fit(y_true.values)
    y_true_num = le.transform(y_true)
    params = {'affinity': ['euclidean', 'cosine'],
              'linkage': ['single', 'complete', 'average'],
              'n_clusters': n_clusters}

    da = hyperparameter_clustering(X, y_true_num, params)
    metric_data, clus, global_time = da
    columns = ['time', 'affinity', 'linkage', 'n_clusters']
    columns = columns + list(INTERNAL_METRICS.keys()) + list(
        EXTERNAL_METRICS.keys())
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'agglo_results_{data_name}')

    return metric_df, global_time


if __name__ == '__main__':
    DATASET_NAME = 'iris.csv'
    N_CLUSTERS = list(range(2, 7, 1))
    METRICS_DF, GLOBAL_TIME = main(DATASET_NAME, N_CLUSTERS)
