from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from clustering.path_definitions import PROCESSED_DATA_PATH


def hyperparameter_clustering(df: pd.DataFrame,
                              parameters: Dict,
                              k: int) -> Tuple[List, List]:
    case_names = []
    case_clustering = []
    for affinity in parameters['affinity']:
        for linkage in parameters['linkage']:
            clustering = AgglomerativeClustering(n_clusters=k,
                                                 affinity=affinity,
                                                 linkage=linkage).fit(df)
            case_clustering.append(clustering)
            case_names.append(f'{affinity}-{linkage}')
    return case_clustering, case_names


p = PROCESSED_DATA_PATH / 'vowel.csv'
vowel_df = pd.read_csv(p,  index_col=0)

params = {'affinity': ['euclidean', 'manhattan', 'cosine'],
          'linkage': ['single', 'complete', 'average', ]}
a, b = hyperparameter_clustering(vowel_df, params, 11)
