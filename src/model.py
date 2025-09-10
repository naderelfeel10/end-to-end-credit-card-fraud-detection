from xgboost import XGBClassifier
import pickle
import pandas as pd

def train_model(X_train, y_train, params: dict):

    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    return xgb_model



def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)