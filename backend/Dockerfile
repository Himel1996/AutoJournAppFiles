FROM python:3.11-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Render sets $PORT at runtime (usually 10000). We listen on that.
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:asgi_app --host 0.0.0.0 --port ${PORT:-10000}"]
