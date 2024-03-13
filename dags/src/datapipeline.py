"""
Modularized Data pipeline to form DAGs in the future
"""
import logging
import os
from dataload import load_data
from zipcode_extract import extract_zipcode
from term_map import map_term
from column_drop import drop_column
from missing_values import handle_missing
from null_drop import drop_null
from credit_year import extract_year
from dummies import get_dummies
from outlier_handle import handle_outliers


if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'src', 'data', 'initial.csv')
    LOADED_DATA_PATH = load_data(excel_path=DEFAULT_EXCEL_PATH)
    EXTRACT_ZIPCODE_PATH = extract_zipcode(input_pickle_path=LOADED_DATA_PATH)
    TERM_MAP_PATH = map_term(input_pickle_path=EXTRACT_ZIPCODE_PATH)
    COLUMN_DROP_PATH = drop_column(input_pickle_path=TERM_MAP_PATH)
    MISSING_VALUES_PATH = handle_missing(input_pickle_path=COLUMN_DROP_PATH)
    NULL_DROP_PATH = drop_null(input_pickle_path=MISSING_VALUES_PATH)
    CREDIT_YEAR_PATH = extract_year(input_pickle_path=NULL_DROP_PATH)
    DUMMIES_PATH = get_dummies(input_pickle_path=CREDIT_YEAR_PATH)
    OUTLIER_HANDLE_PATH = handle_outliers(input_pickle_path=DUMMIES_PATH)