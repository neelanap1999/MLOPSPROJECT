import logging
from src.utils.config import load_config
from src.modules.zipcode_extract import extract_zipcode
from src.modules.term_map import map_term
from src.modules.column_drop import drop_column
from src.modules.missing_values import handle_missing
from src.modules.null_drop import drop_null
from src.modules.credit_year import extract_year
from src.modules.dummies import get_dummies
from src.modules.transform_emp_length import emp_len_transform
from src.modules.outlier_handle import handle_outliers
from src.modules.income_normalization import normalize_amount
from src.modules.labelencode import encode
from src.modules.split import split
from src.modules.scaling_data import scaler
from src.modules.correlation import correlation

def main():
      
    config = load_config()
    Stage_name = "Feature Extraction"

    raw_data = config.feature_extraction.raw_data
    zip_codes = config.feature_extraction.zipcodes
    terms = config.feature_extraction.terms
    column_drop = config.feature_extraction.column_drop
    not_null = config.feature_extraction.not_null
    drop_na = config.feature_extraction.drop_null
    years = config.feature_extraction.years
    dummies = config.feature_extraction.dummies
    emp_len = config.feature_extraction.emp_len
    outlier = config.feature_extraction.outlier
    income_normal = config.feature_extraction.income_normal
    label = config.feature_extraction.label
    x_train = config.feature_extraction.XTRAIN_PATH
    x_test = config.feature_extraction.XTEST_PATH
    y_train = config.feature_extraction.YTRAIN_PATH
    y_test = config.feature_extraction.YTEST_PATH
    scaled_Xtrain = config.feature_extraction.scaled_XTRAIN
    scaled_Xtest = config.feature_extraction.scaled_XTEST

    try:
        logging.info(f"------Stage {Stage_name}------------")

        extract_zipcode( raw_data, zip_codes)
        map_term(zip_codes,  terms)
        drop_column(terms , column_drop)
        handle_missing(column_drop, not_null)
        drop_null(not_null, drop_na)
        extract_year(drop_na, years)
        get_dummies(years, dummies)
        emp_len_transform(dummies, emp_len)
        handle_outliers(emp_len, outlier)
        normalize_amount(outlier, income_normal)
        encode(income_normal, label)
        split(label, x_train, x_test, y_train, y_test) 
        scaler(x_train, x_test, scaled_Xtrain, scaled_Xtest )   
        logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")
        
    except Exception as e:
            print(e)

if __name__=='__main__':
    main()