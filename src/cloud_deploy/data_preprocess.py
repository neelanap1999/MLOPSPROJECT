import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import json 
import gcsfs
import logging
from google.cloud import logging as cloud_logging

# Initialize Google Cloud Logging client
client = cloud_logging.Client()

# Connect the logger to the cloud
client.setup_logging()

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

def stats_to_json(train_data, normalization_stats_gcs_path):
    Stage_name = "Preprocessing data"
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
        try:
            numeric_years = int(years.split()[0])
            return numeric_years
        except ValueError:
            logging.error(f"Unable to map value: {years}. Returning NaN.")
            return np.nan

def preprocess(train_data, output_filepath, normalization_stats_gcs_path, encoder_path):
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

    #save preprocessed data to output_file path
    if not train_data.empty:
        with fs.open(output_filepath, 'w') as f:
            train_data.to_csv(f, index=False)            
    logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")

def update_dataset(train_data, monthly_dataframes, train_data_gcs_path, test_data_gcs_path, preprocessed_train_data_gcs_path, normalization_stats_gcs_path, encoder_path):
    
    sorted_months = sorted(monthly_dataframes.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

    # Attempt to load existing train and test datasets
    try:
        with fs.open(train_data_gcs_path, 'r') as f:
            train_data = pd.read_csv(f)
        with fs.open(test_data_gcs_path, 'r') as f:
            test_data = pd.read_csv(f)
    except FileNotFoundError:
         # If files do not exist, initialize the first two months as training and the third month as testing
        max_m = train_data["issue_d"].max()
        test_data = monthly_dataframes[sorted_months[0]]
        train_data = train_data[train_data["issue_d"] != max_m]
        next_month_index = -1
    else:
        last_test_date  = test_data['issue_d'].max()
        last_test_month = str(last_test_date)

        next_month_index = sorted_months.index(last_test_month) + 1
        
        train_data = pd.concat([train_data, test_data], ignore_index=True)
        test_data = monthly_dataframes[sorted_months[next_month_index]]

    print(train_data.shape, test_data.shape)
     # Save the updated datasets to GCS
    if not train_data.empty:
        with fs.open(train_data_gcs_path, 'w') as f:
            train_data.to_csv(f, index=False)
    if not test_data.empty:
        with fs.open(test_data_gcs_path, 'w') as f:
            test_data.to_csv(f, index=False)
    train_data = train_data.drop("issue_d", axis =1)
    preprocess(train_data, preprocessed_train_data_gcs_path, normalization_stats_gcs_path, encoder_path)


if __name__ == "__main__":
     
    #ouput gcs path to save train_data and test _data
    train_data_gcs_path = "gs://mlops_loan_data/data/train/train_data.csv"
    test_data_gcs_path = "gs://mlops_loan_data/data/test/test_data.csv"

    normalization_stats_gcs_path = "gs://mlops_loan_data/scaler/normalization_stats.json"
    output_train_stats = "gs://mlops_loan_data/scaler/train_schema.txt"

    preprocessed_train_data_gcs_path = "gs://mlops_loan_data/data/train/preprocess/train_data_preprocessed.csv"
    encoder_path = "gs://mlops_loan_data/scaler/"

    # Original dataset
    gcs_loan_data_path = "gs://mlops_loan_data/data/initial_data.csv"
    gcs_loan_test_data_path = "gs://mlops_loan_data/data/test_data.csv"

    train_data = pd.read_csv(gcs_loan_data_path)
    test_data = pd.read_csv(gcs_loan_test_data_path)

    train_data["issue_d"] = pd.to_datetime(train_data["issue_d"]).dt.to_period('M')
    test_data["Year_Month"] = pd.to_datetime(test_data["issue_d"]).dt.to_period('M')
    test_data["issue_d"] = pd.to_datetime(test_data["issue_d"]).dt.to_period('M')
    

    monthly_groups = test_data.groupby('Year_Month')
    monthly_dataframes = {str(period): group.drop('Year_Month', axis=1) for period, group in monthly_groups}


    update_dataset(train_data, monthly_dataframes, train_data_gcs_path, test_data_gcs_path, preprocessed_train_data_gcs_path, normalization_stats_gcs_path, encoder_path)
