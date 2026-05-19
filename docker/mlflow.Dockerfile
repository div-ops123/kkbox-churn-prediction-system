FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.12.0 psycopg2-binary

EXPOSE 5000

# Shell form so $ENV_VARS expand at container runtime, not image build time.
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION}
