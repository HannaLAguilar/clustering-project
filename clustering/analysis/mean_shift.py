import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from clustering.path_definitions import PROCESSED_DATA_PATH

p = PROCESSED_DATA_PATH / 'pen-based.csv'
vowel_df = pd.read_csv(p,  index_col=0)

X = vowel_df.values

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
