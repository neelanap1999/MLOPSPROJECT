
from datetime import datetime
import pytz
import pandas as pd

import joblib
import json

import os
# import pickle
# import plotly.graph_objects as go
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from tabulate import tabulate
from collections import Counter
from sklearn.ensemble import RandomForestClassifier


# Load environment variables


# Initialize variables


def create_model(X_train, y_train):

    
    rfc = RandomForestClassifier(n_estimators=150,max_leaf_nodes=9,max_features=None, max_depth=9)
    rfc.fit(X_train,y_train)            
    
    return  rfc

def save_and_upload_model(model, local_model_path):
  
    joblib.dump(model, local_model_path)


def main():
  
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    XTRAIN_INPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_train_final.parquet')

    YTRAIN_INPUT_PATH=os.path.join(PROJECT_DIR, 'data', 'processed', 'y_train.parquet')

    X_train=pd.read_parquet(XTRAIN_INPUT_PATH)
    y_train=pd.read_parquet(YTRAIN_INPUT_PATH)
    
    # Training the model
    model = create_model(X_train, y_train)
    print("Model Built Successfully!")


    # Save the model locally and upload to GCS
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
  
    local_model_path = os.path.join(PROJECT_DIR, 'data', 'processed', 'model.pkl')

    save_and_upload_model(model, local_model_path)

if __name__ == "__main__":
    main()