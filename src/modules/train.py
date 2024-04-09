
from datetime import datetime
import pytz
import pandas as pd
import joblib
import json
import os
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from src.utils.config import load_config

# Initialize variables
def create_model(X_train, y_train):

    rfc = RandomForestClassifier(n_estimators=150,max_leaf_nodes=9,max_features=None, max_depth=9)
    rfc.fit(X_train,y_train)            
    
    return  rfc

def save_and_upload_model(model, local_model_path):
  
    joblib.dump(model, local_model_path)


