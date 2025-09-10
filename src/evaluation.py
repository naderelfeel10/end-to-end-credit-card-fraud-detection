from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

def evaluate_model(xgb_model, X_test, y_test):

    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba[:,1])
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_threshold = thresholds[f1_scores.argmax()]
    y_pred_custom = (y_proba[:,1] >= best_threshold).astype(int)


    return {
                "best_threshold": best_threshold,
        "precision": precision_score(y_test, y_pred_custom),
        "recall": recall_score(y_test, y_pred_custom),
        "f1": f1_score(y_test, y_pred_custom)
    }