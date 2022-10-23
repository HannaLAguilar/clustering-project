from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_internal_index(df: pd.DataFrame,
                        k_real: int,
                        title: Optional[str] = None):
    n_clusters = df['n_clusters']

    if 'inertia' in df.columns:
        internal_indexs = ['inertia', 'calinski', 'davies', 'silhouette']
        ylbales = ['Inertia', 'Calinski-Harabasz',
                   'Davies&-Bouldin', 'Silhouette']
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    else:
        internal_indexs = ['calinski', 'davies', 'silhouette']
        ylbales = ['Calinski-Harabasz', 'Davies&-Bouldin', 'Silhouette']
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))

    axs = axs.flatten()
    fig.suptitle(title)
    for i, ii_index in enumerate(internal_indexs):
        axs[i].plot(n_clusters,
                    df[ii_index],
                    '--bo',
                    alpha=0.6,
                    markersize=8)
        axs[i].set_xlabel('n cluster')
        axs[i].set_ylabel(f'{ylbales[i]}')
        axs[i].axvline(k_real, c='r', linewidth=2,
                       alpha=0.7, label='True n_clusters')
        axs[i].legend()
    fig.tight_layout()
    return fig


def plot_external_index(df: pd.DataFrame,
                        title: Optional[str] = None):
    external_indexs = ['ARI', 'AMI', 'homo', 'compl', 'v-measure']
    ylabels = ['Adjusted Rand-Index',
               'Adjusted Mutual',
               'Homogeneity',
               'Completeness',
               'V-measure']

    fixed_cols = ['n_clusters']
    selected_columns = fixed_cols + external_indexs
    new_df = df[selected_columns]
    new_df.columns = fixed_cols + ylabels

    fig, ax0 = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(title)
    new_df.plot.bar(x='n_clusters', rot=0, alpha=0.8, ax=ax0)
    ax0.set_xlabel('n cluster')
    ax0.legend(loc='best')
    ax0.set_ylabel('External index')
    fig.tight_layout()

    return fig

