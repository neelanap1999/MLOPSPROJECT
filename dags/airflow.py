import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
import os
from src.dataload import load_data
from src.zipcode_extract import extract_zipcode
from src.term_map import map_term
from src.column_drop import drop_column
from src.missing_values import handle_missing
from src.null_drop import drop_null
from src.credit_year import extract_year
from src.dummies import get_dummies
from src.outlier_handle import handle_outliers
from src.dataload import DEFAULT_PICKLE_PATH

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'src', 'data', 'initial.csv')

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')


# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2024, 3, 4),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Create a DAG instance named 'datapipeline' with the defined default arguments
dag = DAG(
    'datapipeline',
    default_args=default_args,
    description='Airflow DAG for the datapipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    op_kwargs={'pickle_path': DEFAULT_PICKLE_PATH},
    dag=dag,
)

extract_zipcode_task = PythonOperator(
    task_id='extract_zipcode_task',
    python_callable=extract_zipcode,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="load_data_task") }}',
    },
    dag=dag,
)

term_map_task = PythonOperator(
    task_id='term_map_task',
    python_callable=map_term,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="extract_zipcode_task") }}',
    },
    dag=dag,
)

column_drop_task = PythonOperator(
    task_id='column_drop_task',
    python_callable=drop_column,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="term_map_task") }}',
    },
    dag=dag,
)

missing_values_task = PythonOperator(
    task_id='missing_values_task',
    python_callable=handle_missing,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="column_drop_task") }}',
    },
    dag=dag,
)

null_drop_task = PythonOperator(
    task_id='null_drop_task',
    python_callable=drop_null,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="missing_values_task") }}',
    },
    dag=dag,
)

credit_year_task = PythonOperator(
    task_id='credit_year_task',
    python_callable=extract_year,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="null_drop_task") }}',
    },
    dag=dag,
)

dummies_task = PythonOperator(
    task_id='dummies_task',
    python_callable=get_dummies,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="credit_year_task") }}',
    },
    dag=dag,
)

outlier_handle_task = PythonOperator(
    task_id='outlier_handle_task',
    python_callable=handle_outliers,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="dummies_task") }}',
    },
    dag=dag,
)





load_data_task >> extract_zipcode_task >> term_map_task >> column_drop_task >> \
missing_values_task >> null_drop_task >> credit_year_task >> \
      dummies_task >> outlier_handle_task 

logger.info("DAG tasks defined successfully.")

if __name__ == "__main__":
    dag.cli()