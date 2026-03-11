"""
Airflow DAG to run the job scanner every 6 hours.

Usage:
- Place this file in your Airflow DAGs folder (or symlink it).
- Ensure the project (and OPENAI_*/templates) are accessible from the Airflow worker.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from jobs_pipeline import run_job_scan


DEFAULT_ARGS = {
    "owner": "reality-check-ats",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="reality_check_job_scanner",
    default_args=DEFAULT_ARGS,
    description="Scan curated New Grad GitHub repos for new entry-level roles and generate resume suggestions.",
    schedule_interval="0 */6 * * *",  # every 6 hours
    start_date=datetime(2026, 3, 10),
    catchup=False,
    max_active_runs=1,
) as dag:

    def _run_scan() -> None:
        """
        Wrapper to run the job scan. Guardrails:
        - Limits jobs per run (see MAX_JOBS_PER_RUN in jobs_pipeline.py)
        - If there are no new jobs, this is effectively a no-op.
        """
        # Use default model from jobs_pipeline; override here if needed.
        new_count, total = run_job_scan()
        print(f"Job scan complete. New jobs: {new_count}, total stored: {total}")

    run_scan = PythonOperator(
        task_id="run_job_scan",
        python_callable=_run_scan,
    )

