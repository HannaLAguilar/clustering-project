import pandas as pd

from clustering.path_definitions import PROCESSED_DATA_PATH, ROOT_PATH
from clustering.visualization import visualize


PATH_REPORT_FIGURES = ROOT_PATH / 'reports/figures'

if __name__ == '__main__':
    data_name = 'pen-based.csv'
    dataset_name = data_name.split('.')[0]
    n_classes = 10
    df = pd.read_csv(PROCESSED_DATA_PATH / f'agglo_results_{data_name}',
                     index_col=0)

    cases = {'affinity': ['euclidean', 'cosine'],
             'linkage': ['single', 'complete', 'average']}

    for affinity in cases['affinity']:
        for linkage in cases['linkage']:
            case = [affinity, linkage]
            title = f'{case[0].capitalize()} - {case[1]}'
            case_df = df[(df['affinity'] == case[0])
                         & (df['linkage'] == case[1])]
            fig = visualize.plot_internal_index(case_df, n_classes, title)
            fig.savefig(PATH_REPORT_FIGURES /
                        f'{dataset_name}_{case[0]}_{case[1]}.png')
