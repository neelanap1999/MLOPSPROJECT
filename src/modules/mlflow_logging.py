import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Read the wine-quality csv file from the URL
    XTRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_train_final.parquet')
    YTRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','y_train.parquet')
    XTEST_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_test_final.parquet')
    YTEST_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','y_test.parquet')
    # Split the data into training and test sets. (0.75, 0.25) split.
    
    X_train=pd.read_parquet(XTRAIN_PATH)
    y_train=pd.read_parquet(YTRAIN_PATH)
    X_test=pd.read_parquet(XTEST_PATH)
    y_test=pd.read_parquet(YTEST_PATH)
    # The predicted column is "quality" which is a scalar from [3, 9]

    n_estimators = [25, 50, 100, 150]
    max_features = ['sqrt', 'log2', None]
    max_depth = [3, 6, 9]
    max_leaf_nodes = [3, 6, 9]
    for a in n_estimators:
        for b in max_features:
            for c in max_depth:
                for d in max_leaf_nodes:

                    with mlflow.start_run(run_name='RandomForestClassifier'):
                        lr = RandomForestClassifier(n_estimators=a, max_features=b, max_depth=c, max_leaf_nodes=d, random_state=42)
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_test)
                        print(f"  Accuracy: {accuracy_score(y_test,y_pred)}")
                        print(f"  Roc_auc_score: {roc_auc_score(y_test,y_pred)}")

                        mlflow.log_param('n_estimators',a)
                        mlflow.log_param('max_features',b)
                        mlflow.log_param('max_depth',c)
                        mlflow.log_param('max_leaf_nodes',d)

                        mlflow.log_metric("accuracy", accuracy_score(y_test,y_pred))
                        mlflow.log_metric('roc_auc_score',roc_auc_score(y_test,y_pred))

                        predictions = lr.predict(X_train)
                        signature = infer_signature(X_train, predictions)
                        predictions = lr.predict(X_train)
                        signature = infer_signature(X_train, predictions)

                        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                        
                        # Model registry does not work with file store
                        if tracking_url_type_store != "file":
                            # Register the model
                            # There are other ways to use the Model Registry, which depends on the use case,
                            # please refer to the doc for more information:
                            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                            mlflow.sklearn.log_model(
                                lr, "model", registered_model_name="RandomForesrClassifier", signature=signature
                            )
                        else:
                            mlflow.sklearn.log_model(lr, "model", signature=signature)