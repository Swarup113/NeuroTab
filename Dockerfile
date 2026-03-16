FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app:app"]