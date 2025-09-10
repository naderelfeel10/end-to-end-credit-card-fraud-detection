import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

def preprocessing(data: pd.DataFrame, config: dict):

    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=config["data"]["test_size"],random_state=config["data"]["random_state"],stratify=y)
    smote_tomek = SMOTETomek(sampling_strategy=config["data"]["sampling_strategy"],random_state=config["data"]["random_state"])
    X_sm_tomek, y_sm_tomek = smote_tomek.fit_resample(X_train, y_train)

    return X_sm_tomek,y_sm_tomek,X_test,y_test
