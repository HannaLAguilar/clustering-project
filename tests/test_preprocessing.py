import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from clustering.get_data import preprocessing
from clustering.path_definitions import RAW_DATA_PATH, PROCESSED_DATA_PATH


def test_numerical_dataset():
    data_name = 'pen-based.arff'
    p = RAW_DATA_PATH / data_name
    df, _ = preprocessing.import_data(p)
    df = df.iloc[:, :-1]
    assert df.shape == (10992, 16)
    assert pd.notna(df).sum().all()
    X = df.values
    X_trans = StandardScaler().fit_transform(X)
    data_name_csv = f'{data_name.split(".")[0]}.csv'
    df_trans = pd.read_csv(PROCESSED_DATA_PATH / data_name_csv, index_col=0)
    np.testing.assert_allclose(df_trans.iloc[:, :-1].values, X_trans)


def test_numerical_categorical_dataset():
    data_name = 'vowel.arff'
    p = RAW_DATA_PATH / data_name
    df, _ = preprocessing.import_data(p)
    df = df.iloc[:, :-1]
    assert df.shape == (990, 13)
    assert pd.notna(df).sum().all()

    # Test numerical data
    numerical_features = df.select_dtypes(include=np.number).columns
    assert numerical_features.shape[0] == 10
    X = df[numerical_features].values
    X_trans = StandardScaler().fit_transform(X)
    data_name_csv = f'{data_name.split(".")[0]}.csv'
    df_trans = pd.read_csv(PROCESSED_DATA_PATH / data_name_csv, index_col=0)
    np.testing.assert_allclose(df_trans[numerical_features].values, X_trans)

    # Test categorical data
    cat_features = df.columns.difference(numerical_features)
    assert cat_features.shape[0] == 3
    df_cat = df[cat_features]
    X = df_cat.values
    X_trans = OneHotEncoder().fit_transform(X).toarray()
    assert X_trans.shape[1] == df_cat.nunique().sum()
    df_trans_cat = df_trans[df_trans.columns.difference(numerical_features)]
    np.testing.assert_allclose(df_trans_cat.iloc[:, :-1].values, X_trans)






