# Use the official Python slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app/env

# Install extra packages this env needs (curl for healthchecks, basic fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-unifont \
    fonts-noto-color-emoji \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files first (for better caching)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/env

# Expose ports
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server directly using uvicorn 
# (Replaces start.sh from the template since this is a pure FastAPI app)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
