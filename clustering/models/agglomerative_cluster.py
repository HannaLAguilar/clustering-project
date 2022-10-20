from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from time import time

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from clustering.path_definitions import PROCESSED_DATA_PATH
from clustering.visualization import visualize


def hyperparameter_clustering(df: pd.DataFrame,
                              parameters: Dict):
    t0 = time()
    # Metrics
    internal_metrics = {'calinski': metrics.calinski_harabasz_score,
                        'davies': metrics.davies_bouldin_score,
                        'silhouette': metrics.silhouette_score}

    # Perform clustering
    data = []
    agglo_clustering = []
    for affinity in parameters['affinity']:
        for linkage in parameters['linkage']:
            for k in parameters['n_clusters']:
                for method_name, method in internal_metrics.items():
                    result = {}
                    clustering = AgglomerativeClustering(n_clusters=k,
                                                         affinity=affinity,
                                                         linkage=linkage).fit(
                        df.values)
                    agglo_clustering.append(clustering)
                    index_value = method(df.values, clustering.labels_)
                    result['affinity'] = affinity
                    result['linkage'] = linkage
                    result['n_cluster'] = k
                    result['internal_index'] = method_name
                    result['ii_value'] = index_value
                    data.append(result)

    func_time = time() - t0
    return data, agglo_clustering, func_time


p = PROCESSED_DATA_PATH / 'vowel.csv'
vowel_df = pd.read_csv(p, index_col=0)
n_clusters = list(range(2, 16, 1))
params = {'affinity': ['euclidean', 'cosine'],
          'linkage': ['single', 'complete', 'average'],
          'n_clusters': n_clusters}

metric_data, clus, time_perform = hyperparameter_clustering(vowel_df, params)
metric_data_df = pd.DataFrame(metric_data)

case = ['euclidean', 'average', 'silhouette']
title = f'{case[0]}-{case[1]}'
aa = metric_data_df[(metric_data_df.affinity == case[0]) &
                    (metric_data_df.linkage == case[1]) &
                    (metric_data_df.internal_index == case[2])]

visualize.plot_internal_index(n_clusters,
                              aa.ii_value,
                              title,
                              case[2])
