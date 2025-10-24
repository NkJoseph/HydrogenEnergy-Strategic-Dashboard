# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies with retry logic
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p pretrained-models/global pretrained-models/regional \
    processed_data dataset

# Set permissions
RUN chmod +x start_api.py

# Expose ports
EXPOSE 8000 8050

# Create a startup script that runs both services
RUN echo '#!/bin/bash\n\
# Start API in background\n\
python start_api.py &\n\
API_PID=$!\n\
\n\
# Wait a moment for API to start\n\
sleep 5\n\
\n\
# Start Dashboard\n\
python dashboard.py &\n\
DASH_PID=$!\n\
\n\
# Wait for both processes\n\
wait $API_PID $DASH_PID' > start_services.sh && \
chmod +x start_services.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["./start_services.sh"]
