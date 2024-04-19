import pandas as pd
import tensorflow_data_validation as tfdv
import gcsfs
import io

def read_data_from_gcp(bucket_name, file_prefix):
    """Read data from GCP bucket using gcsfs."""
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(f'{bucket_name}/{file_prefix}*')
    data = []
    for file in files:
        with fs.open(file, 'rb') as f:
            df = pd.read_csv(io.BytesIO(f.read()))
            data.append(df)
    return pd.concat(data)

def generate_train_schema(train_data_df, output_train_stats):
    """Generate training schema."""
    train_stats = tfdv.generate_statistics_from_dataframe(train_data_df)
    tfdv.write_stats_text(train_stats, output_train_stats)

def validate_schema(train_stats_file, test_df):
    """Validate schema against test data."""
    train_stats = tfdv.load_statistics(train_stats_file)
    test_stats = tfdv.generate_statistics_from_dataframe(test_df)
    anomalies = tfdv.validate_statistics(test_stats, schema=train_stats.schema)
    tfdv.write_anomalies_text(anomalies, 'anomalies.txt')

def main():
    bucket_name = 'mlops_loan_data/data/'

    train_data_df = read_data_from_gcp(bucket_name, 'train/')
    test_data_df = read_data_from_gcp(bucket_name, 'test/')

    output_train_stats = 'train_stats.txt'
    generate_train_schema(train_data_df, output_train_stats)

    validate_schema(output_train_stats, test_data_df)

if __name__ == "__main__":
    main()
