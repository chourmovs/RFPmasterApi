FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    WORKSPACE=/opt/workspace \
    API_DIR=/app \
    CORE_DIR=/opt/workspace/RFPmaster \
    PYTHONPATH=/app:/opt/workspace/RFPmaster

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

ARG GITHUB_REPO_CORE=chourmovs/RFPmaster.git
ARG BRANCH_CORE=main
ARG GITHUB_TOKEN

RUN test -n "$GITHUB_TOKEN" \
    && mkdir -p /opt/workspace \
    && git clone --depth=1 --branch "${BRANCH_CORE}" \
       "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO_CORE}" \
       "${CORE_DIR}"

RUN if [ -f /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi \
    && if [ -f "${CORE_DIR}/requirements.txt" ]; then pip install --no-cache-dir -r "${CORE_DIR}/requirements.txt"; fi

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "rfp_api_app:app", "--host", "0.0.0.0", "--port", "8000"]
