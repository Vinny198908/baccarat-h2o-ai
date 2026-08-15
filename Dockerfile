FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV H2O_MAX_MEM=1G
ENV H2O_MAX_MODELS=10
ENV H2O_MAX_RUNTIME_SECS=120

CMD ["python", "h2o_baccarat_server_FINAL_SERVE_INDEX.py"]
