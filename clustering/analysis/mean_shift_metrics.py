from time import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import MeanShift, estimate_bandwidth

from clustering.analysis import definitions
from clustering.path_definitions import PROCESSED_DATA_PATH, ROOT_PATH
from clustering.visualization import visualize


def hyperparameter_clustering(X: np.ndarray,
                              y_true: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              definitions.INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              definitions.EXTERNAL_METRICS.values()) -> \
        Tuple[List, float]:

    t0 = time()
    da = []

    for quantile in parameters['quantile']:
        # Perform clustering
        t0 = time()
        bandwidth = estimate_bandwidth(X,
                                       quantile=quantile,
                                       n_samples=500)
        try:

            clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            clustering.fit(X)
            tf = time() - t0

            # Get n_cluster
            labels_unique = np.unique(clustering.labels_)
            n_clusters_ = len(labels_unique)

            # Save in a list
            result = [tf, quantile, n_clusters_]

            # Internal index metrics
            if n_clusters_ == 1:
                result += [np.nan] * 8
            else:
                result += [m(X, clustering.labels_)
                           for m in internal_metrics]
                # External index metrics
                result += [m(y_true, clustering.labels_)
                           for m in external_metrics]
            da.append(result)
        except Exception as e:
            print(e)

    func_time = time() - t0
    return da, func_time


def get_metric_dataset(data_name: str, save: bool = True):
    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1]
    y_true = df['y_true']

    # Parameters for cluster
    params = {'quantile': [0.2, 0.3, 0.4, 0.5]}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true, params)
    metric_data, global_time = da
    columns = ['time', 'quantile', 'n_clusters']
    columns = columns + list(definitions.INTERNAL_METRICS.keys()) + list(
        definitions.EXTERNAL_METRICS.keys())

    # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH
                         / f'mean_shift_results_{data_name}')

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
        title = 'mean_shift'
        raw_name = dataset_name.split('.')[0]

        """
        Dataset are not good enough for plots. 
        We will explained in the report.
        """




