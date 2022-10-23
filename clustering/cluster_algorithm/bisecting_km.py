import numpy as np
import pandas as pd

from clustering.cluster_algorithm.kmeans import kmeans, squared_dist, Results


def inertia_per_cluster(X: np.ndarray,
                        labels: np.ndarray,
                        centers: np.ndarray,
                        k: int = 2) -> np.ndarray:
    cluster_inertia = [0] * k
    for x, label in zip(X, labels):
        cluster_inertia[label] += squared_dist(x
                                               , centers[label])
    return np.array(cluster_inertia)


def join_data_labels(da, i):
    df = da[f'cluster_{i}']
    n_features = len(df)
    labels = np.ones(n_features) * i
    df['labels'] = labels
    return df


def run_bkm(n_clusters: int,
            X_df: pd.DataFrame):
    running_n_cluster = 1
    da = {}

    while running_n_cluster != n_clusters:
        # Kmeans with k=2
        results = kmeans(2, X_df.values)

        # Calculate inertia per each cluster
        cluster_inertia = inertia_per_cluster(X_df.values,
                                              results.labels,
                                              results.centers)

        # Chose the cluster with the biggest inertia and split it
        chosen_cluster = np.argmax(cluster_inertia, axis=0)
        chosen_cluster_data = X_df[results.labels == chosen_cluster]

        # Remain cluster
        _name = f'cluster_{running_n_cluster - 1}'
        da[_name] = X_df[results.labels != chosen_cluster]

        # Increase
        X_df = chosen_cluster_data
        running_n_cluster += 1

        if running_n_cluster == n_clusters:
            _name = f'cluster_{running_n_cluster - 1}'
            da[_name] = X_df

    return da


def dataset_sorted(da):
    dfs = pd.concat([join_data_labels(da, i) for i in range(len(da))])
    dfs.sort_index(inplace=True)
    return dfs


def bisecting_kmeans(n_clusters: int,
                     X_df: pd.DataFrame):
    da = run_bkm(n_clusters, X_df)
    final_df = dataset_sorted(da)
    X = final_df.iloc[:, :-1].values
    labels = final_df['labels']
    centers = final_df.groupby('labels').mean()

    inertia = 0
    for x, label in zip(X, labels):
        inertia += squared_dist(x,
                                centers.iloc[int(label), :].values)

    return Results(labels=final_df.labels,
                   centers=centers,
                   inertia=inertia)
