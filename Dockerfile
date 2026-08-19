# ---- Stage 1: Build ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY main.py .

# Create documents directory (mount point for PDF files)
RUN mkdir -p /app/documents

# Environment variables (override at runtime via docker run -e or .env)
ENV PYTHONUNBUFFERED=1
ENV DB_HOST=host.docker.internal
ENV DB_PORT=5432
ENV DB_NAME=postgres
ENV DB_USER=postgres
ENV DB_PASSWORD=""
ENV GEMINI_API_KEY=""

# Expose no ports (CLI application)
# For future API server, uncomment: EXPOSE 8000

ENTRYPOINT ["python", "main.py"]
