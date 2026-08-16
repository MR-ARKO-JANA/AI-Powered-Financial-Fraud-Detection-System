# Production Dockerfile for FinGuard AI
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user for security
RUN groupadd -r finguard && useradd -r -g finguard -d /app -s /sbin/nologin finguard \
    && mkdir -p /app/data \
    && chown -R finguard:finguard /app

USER finguard

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default launch script
CMD ["bash", "run.sh"]
