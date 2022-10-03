from typing import Tuple
import pandas as pd
from pathlib import Path
from scipy.io import arff

from clustering.path_definitions import RAW_DATA_PATH

import cv2


def import_data(p: Path) -> Tuple[pd.DataFrame,
                                  arff.arffread.MetaData]:
    data, meta = arff.loadarff(p)
    df = pd.DataFrame(data)
    return df, meta


def check_null_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


# Explore
path_file = RAW_DATA_PATH / 'audiology.arff'
df, meta = import_data(path_file)
nulls = check_null_values(df)
