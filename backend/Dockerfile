FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed, minimal)
RUN apt-get update && apt-get install -y \
    build-essential

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose the port Render provides
ENV PORT=10000
EXPOSE 10000

# Start the FastAPI app via uvicorn
CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "10000"]
