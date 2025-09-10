import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocessing
from src.utils import load_config
def test_preprocessing_shapes():

    df = pd.read_csv('data/creditcard.csv')
    config = load_config()

    X_train, y_train, X_test, y_test = preprocessing(df, config)


    # testing if shapes is not correct
    assert X_train.shape[0] > 0, "X_train is  empty"
    assert X_test.shape[0] > 0, "X_test is empty "

    
    # checking missing columns :  
    expected_cols = X_train.columns
    for col in expected_cols:
        assert col in X_train.columns, f"{col} missing in X_train"

    # Test labels
    assert set(y_train.unique()).issubset({0, 1}), "y_train contains invalid values"
    assert set(y_test.unique()).issubset({0, 1}), "y_test contains invalid values"
