import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os
from datetime import datetime

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

XTRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_train_final.parquet')
XTEST_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_train_final.parquet')

YTRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_train_final.parquet')
YTEST_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_train_final.parquet')



'''def eval_metrics(actual, pred):
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy'''


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    X_train=pd.read_parquet(XTRAIN_PATH)
    X_test=pd.read_parquet(XTEST_PATH)
    y_train=pd.read_parquet(YTRAIN_PATH)
    y_test=pd.read_parquet(YTEST_PATH)

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    for a in n_estimators:
        for b in max_features:
            for c in max_depth:
                for d in min_samples_split:
                    for e in min_samples_leaf:
                        for f in bootstrap:
                            with mlflow.start_run() as run:
                                run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
                                # Apply KMeans clustering using the optimal k
                                rfc = RandomForestClassifier(n_estimators=a,max_features=b,max_depth=c,min_samples_split=d,min_samples_leaf=e,bootstrap=f)
                                rfc.fit(X_train,y_train)
                                y_pred=rfc.predict(X_test)

                                mlflow.log_param("n_estimators", a)
                                
                                mlflow.log_param("max_features", b)
                             
                                mlflow.log_param("max_depth", c)
                              
                                mlflow.log_param("min_samples_split", d)
                           
                                mlflow.log_param("min_samples_leaf", e)
                                
                                mlflow.log_param("bootstrap", f)
                           


                                accuracy=accuracy_score(y_test, y_pred)
                                mlflow.log_metric("accuracy_score", accuracy)
                                print(accuracy)

                                # Log the model
                                
                                signature = infer_signature(X_train, y_pred)
                                mlflow.sklearn.log_model(rfc, "model", signature=signature)
                                run_dict={}

                                run_id = run.info.run_id
                                print(f"Run ID: {run_id}")
                                run_dict[f"{run_name}"] = run_id

                                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                                if tracking_url_type_store != "file":
                                    # Register the model
                                    # There are other ways to use the Model Registry, which depends on the use case,
                                    # please refer to the doc for more information:
                                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                                    mlflow.sklearn.log_model(
                                        rfc, "model", registered_model_name="RandomForestClassifier", signature=signature
                                    )
                                else:
                                    mlflow.sklearn.log_model(rfc, "model", signature=signature)