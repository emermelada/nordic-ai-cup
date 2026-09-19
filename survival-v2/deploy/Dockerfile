# Survival v2 agent endpoint as a container (drop-in for any Linux VPS with Docker).
#
#   docker build -f deploy/Dockerfile -t surv-v2 .          # run from survival-v2/
#   docker run -d --name surv-v2 -p 9052:9052 --restart unless-stopped surv-v2
#   curl -s localhost:9052/api                              # {"message": "Agent endpoint running!", ...}
#
# Submit http://<host>:9052/predict (or put it behind an existing HTTPS proxy on the same path).
FROM python:3.12-slim
WORKDIR /app
COPY deploy/requirements-server.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY hive.py server.py /app/
ENV SURV_LOG=/app/server_log.jsonl SURV_TRACE=/app/server_trace.csv
EXPOSE 9052
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9052", "--loop", "uvloop", "--http", "httptools", "--no-access-log"]
