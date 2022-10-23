from typing import List, Optional, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_internal_index(df: pd.DataFrame,
                        k_real: int,
                        title: Optional[str] = None):
    n_clusters = df['n_clusters']
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
        axs[i].set_ylabel(f'{ylbales[i]} index')
        axs[i].axvline(k_real, c='r', linewidth=2,
                       alpha=0.7, label='True n_clusters')
        axs[i].legend()
    fig.tight_layout()
    return fig


def plot_inertial():
    pass


def plot_external_index():
    pass

