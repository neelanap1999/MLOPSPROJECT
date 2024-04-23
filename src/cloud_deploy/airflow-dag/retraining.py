import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator

LOCAL_PREPROCESS_FILE_PATH = '/tmp/preprocess.py'
GITHUB_PREPROCESS_RAW_URL = 'https://raw.githubusercontent.com/neelanap1999/MLOPSPROJECT/main/src/cloud_deploy/data_preprocess.py'  # Adjust the path accordingly

LOCAL_TRAIN_FILE_PATH = '/tmp/train.py'
GITHUB_TRAIN_RAW_URL = 'https://raw.githubusercontent.com/neelanap1999/MLOPSPROJECT/main/src/cloud_deploy/trainer/train.py'  # Adjust the path accordingly

default_args = {
    'owner': 'MLOPS_CREDIT_ASSESMENT',
    'start_date': dt.datetime(2024, 4, 21),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='naveensvs.us@gmail.com',
        subject='Success Notification from Airflow',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to='naveensvs.us@gmail.com',
        subject='Failure Notification from Airflow',
        html_content='<p>The task failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Model retraining at every 1 hours',
    schedule_interval='@hourly',  # Every 12 hours
    catchup=False,
)

# Defining the email task to send notification
send_email = EmailOperator(
    task_id='send_email',
    to='naveensvs.us@gmail.com',    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow.</p>',
    dag=dag,
    on_failure_callback=notify_failure,
    on_success_callback=notify_success
)

# Tasks for pulling scripts from GitHub
pull_preprocess_script = BashOperator(
    task_id='pull_preprocess_script',
    bash_command=f'curl -o {LOCAL_PREPROCESS_FILE_PATH} {GITHUB_PREPROCESS_RAW_URL}',
    dag=dag,
)

pull_train_script = BashOperator(
    task_id='pull_train_script',
    bash_command=f'curl -o {LOCAL_TRAIN_FILE_PATH} {GITHUB_TRAIN_RAW_URL}',
    dag=dag,
)


env = {
    'AIP_STORAGE_URI': 'gs://mlops_loan_data/model'
}

# Tasks for running scripts
run_preprocess_script = BashOperator(
    task_id='run_preprocess_script',
    bash_command=f'python {LOCAL_PREPROCESS_FILE_PATH}',
    env=env,
    dag=dag,
)

run_train_script = BashOperator(
    task_id='run_train_script',
    bash_command=f'python {LOCAL_TRAIN_FILE_PATH}',
    env=env,
    dag=dag,
)

# Setting up dependencies
pull_preprocess_script >> pull_train_script >> run_preprocess_script >> run_train_script >> send_email 