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

ARG GITHUB_REPO_CORE_URL=https://github.com/chourmovs/RFPmaster.git
ARG BRANCH_CORE=main

RUN mkdir -p /opt/workspace \
    && git clone --depth=1 --branch "${BRANCH_CORE}" "${GITHUB_REPO_CORE_URL}" "${CORE_DIR}"

RUN if [ -f /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi \
    && if [ -f "${CORE_DIR}/requirements.txt" ]; then pip install --no-cache-dir -r "${CORE_DIR}/requirements.txt;"; fi
