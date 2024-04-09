import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
import os
from src.download_data import ingest_data
from src.dataload import load_data
from src.zipcode_extract import extract_zipcode
from src.term_map import map_term
from src.column_drop import drop_column
from src.missing_values import handle_missing
from src.null_drop import drop_null
from src.credit_year import extract_year
from src.dummies import get_dummies
from src.outlier_handle import handle_outliers
from src.income_normalization import normalize_amount
from src.transform_emp_length import emp_len_transform
from src.scaling_data import scaler
from src.correlation import correlation
from src.pca import analyze_pca
from src.labelencode import encode
from src.split import split
from src.perform_tfdv import validate_data_tfdv
from src.dataload import DEFAULT_PICKLE_PATH
from src.download_data import ingest_data
from airflow.operators.email_operator import EmailOperator

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'src', 'data', 'initial.csv')

# Enabling pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Defining the default arguments for DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2024, 3, 4),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='guptavishesh264@gmail.com',
        subject='Success Notification from Airflow',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)
 
def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to='guptavishesh264@gmail.com',
        subject='Failure Notification from Airflow',
        html_content='<p>The task failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)

# Creating a DAG instance named 'datapipeline' with the defined default arguments
dag = DAG(
    'datapipeline',
    default_args=default_args,
    description='Airflow DAG for the datapipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

# Defining the email task to send notification
send_email = EmailOperator(
    task_id='send_email',
    to='guptavishesh264@gmail.com',    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow.</p>',
    dag=dag,
    on_failure_callback=notify_failure,
    on_success_callback=notify_success
)

# Task to download data from source, calls the 'ingest_data' Python function
ingest_data_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data,
    op_args=["https://drive.google.com/file/d/1NAn7I7iJGxy2AhrmfkVdo37GY1dtGLzw/view?usp=sharing"],
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    op_kwargs={'excel_path': ingest_data_task.output},
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

emp_len_task = PythonOperator(
    task_id='emp_len_task',
    python_callable=emp_len_transform,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="dummies_task") }}',
    },
    dag=dag,
)

outlier_handle_task = PythonOperator(
    task_id='outlier_handle_task',
    python_callable=handle_outliers,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="emp_len_task") }}',
    },
    dag=dag,
)

income_normalize_task = PythonOperator(
    task_id='income_normalize_task',
    python_callable=normalize_amount,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="outlier_handle_task") }}',
    },
    dag=dag,
)

encode_task = PythonOperator(
    task_id='encode_task',
    python_callable=encode,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="income_normalize_task") }}',
    },
    dag=dag,
)

split_task= PythonOperator(
    task_id='split_task',
    python_callable=split,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="encode_task") }}',
    },
    dag=dag,
)

perform_tfdv_task = PythonOperator(
    task_id='perform_tfdv_task',
    python_callable=validate_data_tfdv,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="split_task") }}',
    },
    dag=dag,
)


scaler_task = PythonOperator(
    task_id='scaler_task',
    python_callable=scaler,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="split_task") }}',
    },
    dag=dag,
)

correlation_task = PythonOperator(
    task_id='correlation_task',
    python_callable=correlation,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="scaler_task") }}',
    },
    dag=dag,
)



'''
analyze_pca_task = PythonOperator(
    task_id='analyze_pca_task',
    python_callable=analyze_pca,
    op_kwargs={
        'input_pickle_path': '{{ ti.xcom_pull(task_ids="scaler_task") }}',
    },
    dag=dag,
)
'''


ingest_data_task >> load_data_task >> extract_zipcode_task >> term_map_task >> column_drop_task >> \
missing_values_task >> null_drop_task >> credit_year_task >> \
    dummies_task >> emp_len_task >> outlier_handle_task >> income_normalize_task >> encode_task \
    >> split_task >> perform_tfdv_task >> scaler_task >> correlation_task >> send_email 

logger.info("DAG tasks defined successfully.")

if __name__ == "__main__":
    dag.cli()