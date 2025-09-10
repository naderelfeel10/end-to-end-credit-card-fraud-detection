import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , precision_score , recall_score ,  make_scorer,f1_score , precision_recall_curve , roc_auc_score 
from src.preprocessing import preprocessing
import pickle
import yaml


def train_model(config_path: str = "config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["input_path"]
    model_path = config["model"]["output_path"]
    model_params = config["model"]["params"]


    data = pd.read_csv(data_path)
    X_train,y_train,X_test,y_test = preprocessing(data)

    xgb_model = XGBClassifier(**model_params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)
    print("XGB results : \n")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba[:,1])
    print("best threshold for max f1_score : \n")
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_threshold = thresholds[f1_scores.argmax()]
    print("best Threshold:", best_threshold)
    y_pred_custom = (y_proba[:,1] >= best_threshold).astype(int)
    print(f"Precision : {precision_score(y_test, y_pred_custom)}")
    print(f"Recall : {recall_score(y_test, y_pred_custom)}")
    print(f"F1-score : {f1_score(y_test, y_pred_custom)}")


    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)



if __name__ == "__main__":
    train_model("config/config.yaml")
