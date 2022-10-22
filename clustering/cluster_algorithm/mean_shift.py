import numpy as np
import pandas as pd
import itertools

import sys
sys.path.append('C:/Users/mario/Github/UPC/IML - Introduction to Machine Learning/Work 1 - Clustering Excercise/')


from sklearn.cluster import MeanShift, estimate_bandwidth
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


def hyperparameter_clustering(X: np.ndarray,
                              y_true_num: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              EXTERNAL_METRICS.values()) -> \
        Tuple[List, List, float]:

    t0 = time()
    MS_clustering = []
    da = []
    params_combs = list(itertools.product(*list(parameters.values())))
    param = {}
    for param_comb in params_combs:
        for index, p in enumerate(list(parameters.keys())):
            param[p] = param_comb[index]
        t0 = time()

        # Perform clustering
        bandwidth = estimate_bandwidth(X, quantile=param['quantile'], n_samples=param['n_samples'], random_state=param['random_state'], n_jobs=param['n_jobs'])

        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=param['bin_seeding'], n_jobs=param['n_jobs'])
        clustering.fit(X)
        labels = clustering.labels_
        tf = time() - t0

        print('Bandwidth: ', bandwidth)
        print('Clusters_centers: ', clustering.cluster_centers_)
        print('Labels: ', clustering.labels_)
        print('N_iter: ', clustering.n_iter_)
        print('N_Features: ', clustering.n_features_in_)

        # MS_clustering.append(clustering.labels_)
        # Save in a list
        # result = [clustering,tf, *param_comb]
        # Internal index metrics
        # result += [m(X, clustering.labels_)
        #            for m in internal_metrics]
        # # External index metrics
        # result += [m(y_true_num, clustering.labels_)
        #            for m in external_metrics]
        # da.append(result)

    func_time = time() - t0
    return da, MS_clustering, func_time


def main(data_name: str, save: bool = True):
    # Data
    # path = PROCESSED_DATA_PATH / data_name
    # df = pd.read_csv(path, index_col=0)
    df, _ = import_data('../../data/raw/pen-based.arff')
    X = df.iloc[:, :-1]
    # y_true = df['y_true']
    y_true = df['a17']
    le = LabelEncoder().fit(y_true.values)
    y_true_num = le.transform(y_true)

    # Parameters for cluster
    params = {'quantile': [0.5],
              'n_samples': [X.shape[0]],
              'random_state': [None],
              'n_jobs':[None],
              'bin_seeding':[False]}
    print('X_shape: ',X.shape)

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true_num, params)
    # metric_data, clus, global_time = da
    # columns = ['time', 'affinity', 'linkage', 'n_clusters']
    # columns = columns + list(INTERNAL_METRICS.keys()) + list(
    #     EXTERNAL_METRICS.keys())
    #
    # # Metric dataset
    # metric_df = pd.DataFrame(metric_data, columns=columns)
    #
    # if save:
    #     metric_df.to_csv(PROCESSED_DATA_PATH / f'agglo_results_{data_name}')

    return y_true, le # metric_df, global_time


if __name__ == '__main__':
    DATASET_NAME = 'vowel.csv'
    METRICS_DF, GLOBAL_TIME = main(DATASET_NAME)