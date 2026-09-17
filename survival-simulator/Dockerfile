# 3.12, not 3.14 like the sibling services: pygame (a pinned dep here) publishes no
# prebuilt Linux wheel for cp314, so pip falls back to compiling it from source and
# fails on missing SDL headers. On 3.12 every pinned dep installs as a binary wheel.
FROM python:3.12-slim

# No torch in this service's requirements, so no extra index is needed here.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 9052
CMD ["uvicorn", "agent_server:app", "--host", "0.0.0.0", "--port", "9052"]
