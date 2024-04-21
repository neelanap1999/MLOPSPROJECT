import logging
import pandas as pd
import numpy as np
import pickle
from modules.extract_columns import extract_task
from modules.column_drop import drop_column
from modules.missing_values import handle_missing
from modules.null_drop import drop_null
from modules.dummies import get_dummies
from modules.outlier_handle import handle_outliers
from modules.labelencode import encode
# import tensorflow_data_validation as tfdv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


# import os
import json 
import gcsfs

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

Stage_name = "Feature Extraction"

def stats_to_json(train_data, normalization_stats_gcs_path):
    
    Stage_name = "Preprocessing data"
    # logging.info(f"------Stage {Stage_name}------------")    
    Num_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

    train_data_numeric = train_data[Num_cols]

    # Calculate mean and standard deviation for each feature in the training data
    mean_train = train_data_numeric.mean()
    std_train = train_data_numeric.std()

    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }

    # Save the normalization statistics to a JSON file on GCS
    with fs.open(normalization_stats_gcs_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)
    logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")

def generate_train_schema(train_data_df, output_train_stats):
    """Generate training schema."""
    # train_stats = tfdv.generate_statistics_from_dataframe(train_data_df)
    # tfdv.write_stats_text(train_stats, output_train_stats)

# Define mapping function
def map_years(years):
    """
    Mapping employment length values to numerical values.

    Args:
        years (str or int): The employment length value.

    Returns:
        int or np.nan: The mapped numerical value for employment length, or np.nan if input is NaN.

    Raises:
        None
    """
    if pd.isna(years):  # Handle NaN values
        return np.nan
    elif isinstance(years, int):  # If already an integer, return as is
        return years
    elif years == '< 1 year':
        return 0
    elif years == '10+ years':
        return 10
    else:
        return int(years.split()[0])

def preprocess(train_data, output_filepath, normalization_stats_gcs_path, output_train_stats, encoder_path):
    Stage_name = "Preprocessing data"
    logging.info(f"------Stage {Stage_name}------------")

    # extract zipcodes
    train_data['earliest_cr_line'] = train_data['earliest_cr_line'].apply(lambda x:int(x[-4:]))
    train_data['zipcode'] = train_data['address'].apply(lambda x:str(x[-5:]))
    train_data["zipcode"] = train_data['zipcode'].astype ('category')
    train_data.drop('address',axis=1,inplace=True)

    # encode     
    dic = {' 36 months':36, ' 60 months':60}
    train_data['term'] = train_data.term.map(dic)


    # Applying mapping function to the column
    train_data['emp_length'] = train_data['emp_length'].map(map_years)

    # Label encoding for 'loan_status' column
    le = LabelEncoder()
    train_data['loan_status'] = le.fit_transform(train_data['loan_status'])
    with fs.open(encoder_path + 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    #Columns Drop 
    train_data.drop(['sub_grade','title','emp_title'],axis=1,inplace=True)

    #handle missing values    
    train_data['mort_acc'].fillna(train_data['mort_acc'].mean(),inplace = True)
    train_data['emp_length'].fillna(0,inplace = True)
    train_data['revol_util'].fillna(0,inplace = True)
    train_data['mort_acc'].fillna(0,inplace = True)

    #drop null   
    train_data.dropna(inplace=True)

    # handle_outliers 
    model = IsolationForest(contamination=0.05, random_state=0)
    train_data['Outlier_Scores'] = model.fit_predict(train_data.drop(['loan_status','grade', 'verification_status', 'purpose', 'initial_list_status',
           'application_type', 'home_ownership'],axis=1).to_numpy())
    train_data['Is_Outlier'] = [1 if x == -1 else 0 for x in train_data['Outlier_Scores']]
    train_data = train_data[train_data['Is_Outlier'] == 0].copy()
    train_data.drop(['Is_Outlier','Outlier_Scores'],axis=1,inplace=True)
    
    train_data = train_data.reset_index()
    train_data.drop('index',axis=1,inplace=True)

    #treat Categorical 
    categorical_columns = ['verification_status', 'purpose', 'initial_list_status', 'application_type', 'home_ownership']

    # Use OrdinalEncoder for 'grade' column
    ordinal_encoder = OrdinalEncoder()
    train_data['grade'] = ordinal_encoder.fit_transform(train_data[['grade']])

    # Save the OrdinalEncoder
    with fs.open(encoder_path + 'ordinal_encoder_grade.pkl', 'wb') as f:
        pickle.dump(ordinal_encoder, f)


    one_hot_encoder = OneHotEncoder()
    one_hot_encoded_data = one_hot_encoder.fit_transform(train_data[categorical_columns])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data.toarray(), columns = one_hot_encoder.get_feature_names_out(categorical_columns))
    train_data = pd.concat([train_data.drop(categorical_columns, axis=1), one_hot_encoded_df], axis=1)

    # Save the OneHotEncoder
    with fs.open(encoder_path + 'one_hot_encoder.pkl', 'wb') as f:
        pickle.dump(one_hot_encoder, f)

    #Generate stats on numerical columns only - >  export to JSON
    stats_to_json(train_data, normalization_stats_gcs_path)
    # train_stats = tfdv.generate_statistics_from_dataframe(train_data)
    

    #save preprocessed data to output_file path
    if not train_data.empty:
        with fs.open(output_filepath, 'w') as f:
            train_data.to_csv(f, index=False)            
    print(output_filepath)
    logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")


if __name__ == "__main__":
     
    #ouput gcs path to save train_data and test _data
    train_data_gcs_path = "gs://mlops_loan_data/data/train/train_data.xlsx"
    test_data_gcs_path = "gs://mlops_loan_data/data/train/test_data.xlsx"
    normalization_stats_gcs_path = "gs://mlops_loan_data/scaler/normalization_stats.json"
    output_train_stats = "gs://mlops_loan_data/scaler/train_schema.txt"
    preprocessed_train_data_gcs_path = "gs://mlops_loan_data/data/train/preprocess/train_data_preprocessed.csv"
    encoder_path = "gs://mlops_loan_data/scaler/"

    # Original dataset
    gcs_loan_data_path = "gs://mlops_loan_data/data/initial_data.csv"
    train_data = pd.read_csv(gcs_loan_data_path)
    train_data["issue_d"] = pd.to_datetime(train_data["issue_d"])

    #during data splitting drop issue_d column
    train_data.drop(['issue_d'],axis=1,inplace=True)

    preprocess(train_data, preprocessed_train_data_gcs_path, normalization_stats_gcs_path, output_train_stats, encoder_path)


