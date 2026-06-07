FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY data/06_models/ ./data/06_models/

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/data/06_models/model.pkl

EXPOSE 8000

CMD ["uvicorn", "crash_kedro.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
