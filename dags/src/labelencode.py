import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Configure logging
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'scaler_output.parquet')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'encoder_output.parquet')

def encode(in_path=INPUT_PICKLE_PATH, out_path=OUTPUT_PICKLE_PATH):

    data = pd.read_parquet(in_path)
    le=LabelEncoder()
    data['loan_status']=le.fit_transform(data['loan_status'])
    data.drop(data.columns[0],axis=1)
    data.to_parquet(out_path)

    return out_path
