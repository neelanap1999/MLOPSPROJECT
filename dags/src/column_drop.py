import os
import pickle


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_term.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dropcol.pkl')

def drop_column(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):


    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")

    df.drop(['grade','title','emp_title'],axis=1,inplace=True)

    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path