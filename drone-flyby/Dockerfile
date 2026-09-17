FROM python:3.14-slim

# CPU wheels by default; the GPU compose override passes a CUDA index.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url ${TORCH_INDEX} -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
