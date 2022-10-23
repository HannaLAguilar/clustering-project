from time import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import itertools

from clustering.analysis import definitions
from clustering.path_definitions import PROCESSED_DATA_PATH, ROOT_PATH
from clustering.cluster_algorithm.fcm import FCM
from clustering.visualization import visualize


def hyperparameter_clustering(X: np.ndarray,
                              y_true_num: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              definitions.INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              definitions.EXTERNAL_METRICS.values()) -> \
        Tuple[List, List, float]:

    t0 = time()
    da = []
    params_combs = list(itertools.product(*list(parameters.values())))
    param = {}
    FCM_clustering = []
    for param_comb in params_combs:
        for index, p in enumerate(list(parameters.keys())):
            param[p] = param_comb[index]

        t0 = time()
        # Perform clustering
        centers, labels = FCM(X, param['n_clusters'],
                              param['m'],
                              param['num_iters'],
                              param['term_threshold'],
                              param['limit'])
        tf = time() - t0

        FCM_clustering.append(labels)
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
    return da, FCM_clustering, func_time


def get_metric_dataset(data_name: str,
                       n_clusters: int,
                       save: bool = True):
    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1].values
    y_true = df['y_true']

    # Parameters for cluster
    params = {'n_clusters': n_clusters,
              'm': [2],
              'num_iters': [100],
              'term_threshold': [0.000001],
              'limit': [10]}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true, params)
    metric_data, clus, global_time = da
    columns = ['time', *list(params.keys())]
    columns = columns + list(definitions.INTERNAL_METRICS.keys()) + list(
        definitions.EXTERNAL_METRICS.keys())

    # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'fcm_results_{data_name}')

    return metric_df, global_time


if __name__ == '__main__':

    PATH_REPORT_FIGURES = ROOT_PATH / 'reports/figures'

    DATASET_NAME = ['iris.csv',
                    'vowel.csv',
                    'cmc.csv',
                    'pen-based.csv']

    N_CLUSTERS_RANGE = [definitions.n_clusters_iris,
                        definitions.n_clusters_vowel,
                        definitions.n_clusters_cmc,
                        definitions.n_clusters_pen]

    CLASS_REAL = [definitions.n_iris_real,
                  definitions.n_vowel_real,
                  definitions.n_cmc_real,
                  definitions.n_pen_real]

    for dataset_name, n_clusters, n_real_class in zip(DATASET_NAME,
                                                      N_CLUSTERS_RANGE,
                                                      CLASS_REAL):
        # Main code
        print(f'Processing metrics for {dataset_name} dataset')
        METRIC_DF, GLOBAL_TIME = get_metric_dataset(dataset_name, n_clusters)

        # Plots
        title = 'fcm'
        raw_name = dataset_name.split('.')[0]

        # METRIC_DF = pd.read_csv(PROCESSED_DATA_PATH /
        #                         f'{title}_results_{dataset_name}',
        #                         index_col=0)

        # Internal index
        fig1 = visualize.plot_internal_index(METRIC_DF,
                                             n_real_class,
                                             title)
        fig1.savefig(PATH_REPORT_FIGURES /
                     f'{raw_name}_internal_{title}.png')

        # External index
        fig2 = visualize.plot_external_index(METRIC_DF,
                                             title)
        fig2.savefig(PATH_REPORT_FIGURES /
                     f'{raw_name}_external_{title}.png')
