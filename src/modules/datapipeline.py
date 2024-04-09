"""
Modularized Data pipeline to form DAGs in the future
"""
import logging
import os
from download_data import ingest_data
from dataload import load_data
from zipcode_extract import extract_zipcode
from term_map import map_term
from column_drop import drop_column
from missing_values import handle_missing
from null_drop import drop_null
from credit_year import extract_year
from dummies import get_dummies
from outlier_handle import handle_outliers
from income_normalization import normalize_amount
from transform_emp_length import emp_len_transform
from scaling_data import scaler
from correlation import correlation
from labelencode import encode
from split import split
from pca import analyze_pca
from download_data import ingest_data

""" pipeline for data preprocessing and transformation: Each step takes the output of the previous step as input and saves the processed DataFrame to a pickle file."""
if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_FILE_URL = "https://drive.google.com/file/d/1NAn7I7iJGxy2AhrmfkVdo37GY1dtGLzw/view?usp=sharing"
    #DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'src', 'data', 'initial.csv')
    DOWNLOAD_DATA_PATH = ingest_data(file_url = DEFAULT_FILE_URL)
    LOADED_DATA_PATH = load_data(excel_path=DOWNLOAD_DATA_PATH)
    EXTRACT_ZIPCODE_PATH = extract_zipcode(input_pickle_path=LOADED_DATA_PATH)
    TERM_MAP_PATH = map_term(input_pickle_path=EXTRACT_ZIPCODE_PATH)
    COLUMN_DROP_PATH = drop_column(input_pickle_path=TERM_MAP_PATH)
    MISSING_VALUES_PATH = handle_missing(input_pickle_path=COLUMN_DROP_PATH)
    NULL_DROP_PATH = drop_null(input_pickle_path=MISSING_VALUES_PATH)
    CREDIT_YEAR_PATH = extract_year(input_pickle_path=NULL_DROP_PATH)
    DUMMIES_PATH = get_dummies(input_pickle_path=CREDIT_YEAR_PATH)
    EMP_LEN_PATH = emp_len_transform(input_pickle_path=DUMMIES_PATH)
    OUTLIER_HANDLE_PATH = handle_outliers(input_pickle_path=EMP_LEN_PATH)
    INCOME_NORMAL_PATH = normalize_amount(input_pickle_path=OUTLIER_HANDLE_PATH)
    ENCODE_PATH = encode(input_pickle_path=INCOME_NORMAL_PATH)
    (XTRAIN_PATH,XTEST_PATH,YTRAIN_PATH,YTEST_PATH)=split(input_pickle_path=ENCODE_PATH)
    (XTRAIN_FINAL,XTEST_FINAL) = scaler(xtrain_inpath=XTRAIN_PATH,xtest_inpath=XTEST_PATH)
    CORR_PATH = correlation(input_pickle_path=XTRAIN_FINAL)

    #PCA_PATH = analyze_pca(input_pickle_path=SCALER_PATH)

