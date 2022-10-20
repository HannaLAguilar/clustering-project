from typing import List, Optional, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_internal_index(n_clusters: List,
                        index_values: Union[List, np.ndarray],
                        title: Optional[str],
                        index_name: str):
    plt.figure(),
    plt.title(title)
    plt.plot(n_clusters,
             index_values,
             '--bo',
             alpha=0.6,
             markersize=8)
    plt.xlabel('n cluster')
    plt.ylabel(f'{index_name} index')
