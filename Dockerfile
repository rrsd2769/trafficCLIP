# CPU-only image for the OptimizedTrafficCLIP reproducibility package.
# Datasets are not baked in: mount data/ (and results/ to keep outputs).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpcap0.8 xz-utils curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt requirements-tracking.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-tracking.txt

COPY . .

ENV PYTHONPATH=/work \
    HF_HUB_DISABLE_XET=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

CMD ["python", "-m", "pytest", "tests", "A/tests"]
