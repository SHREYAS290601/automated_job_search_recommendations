#!/usr/bin/env bash
set -euo pipefail

# Root of this project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="reality-check-airflow"
CONTAINER_NAME="reality-check-airflow"

echo "[run_app] Project dir: $PROJECT_DIR"

cd "${PROJECT_DIR}"

echo "[run_app] Installing dev dependencies (pytest) if needed..."
pip show pytest >/dev/null 2>&1 || pip install pytest >/dev/null 2>&1

echo "[run_app] Running tests with pytest..."
if ! pytest; then
  echo "[run_app] Tests failed. Aborting startup."
  exit 1
fi

echo "[run_app] Tests passed."

# Start Airflow container (scheduler+webserver) if not already running.
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    echo "[run_app] Starting existing Airflow container..."
    docker start "${CONTAINER_NAME}" >/dev/null
  else
    echo "[run_app] Building Airflow image..."
    docker build -f "${PROJECT_DIR}/Dockerfile.airflow" -t "${IMAGE_NAME}" "${PROJECT_DIR}"
    echo "[run_app] Running Airflow container..."
    docker run -d \
      --name "${CONTAINER_NAME}" \
      -p 8080:8080 \
      -v "${PROJECT_DIR}:/opt/reality-check-ats" \
      -e AIRFLOW__CORE__DAGS_FOLDER=/opt/reality-check-ats \
      "${IMAGE_NAME}" >/dev/null
  fi
else
  echo "[run_app] Airflow container already running."
fi

echo "[run_app] Starting Streamlit app..."
streamlit run app.py

