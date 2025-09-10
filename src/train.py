import pandas as pd
from utils import load_config, set_seed, get_logger
from preprocessing import preprocessing
from features import add_features
from model import train_model, save_model

from evaluation import evaluate_model

logger = get_logger(__name__)


def run_pipeline():
    config = load_config()

    #print(config)
    set_seed(config["data"]["random_state"])

    logger.info("Loading data...")
    df = pd.read_csv(config["data"]["path"])

    #logger.info("Adding features...") // i will keep it for any future ideas
    #df = add_features(df)

    logger.info("Preprocessing...")
    X_train, y_train, X_test, y_test = preprocessing(df, config)

    logger.info("Training model...")
    model = train_model(X_train, y_train, config["model"]["params"])

    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Evaluation results: {metrics}")

    logger.info("Saving model...")
    save_model(model, config["output"]["model_path"])

if __name__ == "__main__":
    run_pipeline()
