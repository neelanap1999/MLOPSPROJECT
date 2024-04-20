import json
import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score
from src.modules.train import create_model, save_and_upload_model
from src.utils.config import load_config

config = load_config()
Stage_name = "Model training"

YTRAIN_INPUT_PATH = config.feature_extraction.YTRAIN_PATH
YTEST_INPUT_PATH = config.feature_extraction.YTEST_PATH
XTRAIN_INPUT_PATH = config.feature_extraction.scaled_XTRAIN
XTEST_INPUT_PATH = config.feature_extraction.scaled_XTEST
json_file_path = config.training.json_path
local_model_path = config.training.model_path


def main():
        
    try:
        logging.info(f"------Stage {Stage_name}------------")

        X_train = pd.read_parquet(XTRAIN_INPUT_PATH)
        X_test = pd.read_parquet(XTEST_INPUT_PATH)
        y_train = pd.read_parquet(YTRAIN_INPUT_PATH)
        y_test = pd.read_parquet(YTEST_INPUT_PATH)

        # Training the model
        model = create_model(X_train, y_train)
        logging.info("Model Built Successfully!")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)

        scores = {'accuracy': accuracy, 'precision': precision}

        with open(json_file_path,'w') as json_file:
            json.dump(scores,json_file)
        print(f"Scores saved to {json_file_path}.")
        
        # Save the model locally and upload to GCS
        save_and_upload_model(model, local_model_path)
        print(f"Model saved to {local_model_path}.")
        logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")
        
    except Exception as e:
            print(e)

if __name__ == "__main__":
    main()