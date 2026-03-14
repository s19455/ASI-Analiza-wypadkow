FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/crash_kedro/api/ ./api/
COPY data/06_models/model.pkl ./models/model.pkl

ENV MODEL_PATH=/app/models/model.pkl

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
