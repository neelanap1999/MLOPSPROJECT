import pandas as pd
import random
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define function to create sample files
def create_sample_files(original_data, num_samples, num_records_per_sample):
    for i in range(num_samples):
        sample_data = original_data.sample(n=num_records_per_sample)
        '''SAMPLE_PATH = os.path.join(PROJECT_DIR, 'data','sample_test_data_{i+1}.csv')
        sample_data.to_csv(SAMPLE_PATH, index=False)'''
        sample_data.to_csv(f'sample_{i+1}.csv', index=False)
        print(f'Sample test data {i+1} created successfully.')

if __name__ == "__main__":
    original_data = pd.read_csv("test_data.csv")
    # Define parameters
    num_samples = 5
    num_records_per_sample = 1000
    create_sample_files(original_data, num_samples, num_records_per_sample)