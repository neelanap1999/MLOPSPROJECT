import os
import pickle


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dropcol.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_fillna.pkl')

def handle_missing(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):


    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")

    df['mort_acc'].fillna(df['mort_acc'].mean(),inplace = True)
    df['emp_length'].fillna(0,inplace = True)
    df['revol_util'].fillna(0,inplace = True)
    df['mort_acc'].fillna(0,inplace = True)
    
    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path