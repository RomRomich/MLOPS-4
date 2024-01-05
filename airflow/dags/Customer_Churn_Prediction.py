
# Импортируем библиотеки
from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt
 
args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 1, 6),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False
}
 
with DAG(
    dag_id='Customer_Churn_Prediction',
    default_args=args,
    schedule_interval=None,
    tags=['Customer_Churn_Prediction', 'accuracy'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 /scripts/get_data.py",
                            dag=dag)
    process_data = BashOperator(task_id='process_data',
                            bash_command="python3 /scripts/process_data.py",
                            dag=dag)
    train_test_split_data = BashOperator(task_id='train_test_split_data',
                            bash_command="python3 /scripts/train_test_split_data.py",
                            dag=dag)  
    train_model = BashOperator(task_id='train_model',
                            bash_command="python3 /scripts/train_model.py",
                            dag=dag)
    test_model = BashOperator(task_id='test_model',
                            bash_command="python3 /scripts/test_model.py",
                            dag=dag)
    optimization_gyperparametr = BashOperator(task_id='optimization_gyperparametr',
                            bash_command="python3 /scripts/optimization_gyperparametr.py",
                            dag=dag)
    get_data >> process_data >> train_test_split_data >>  train_model  >> optimization_gyperparametr >> test_model
    #
