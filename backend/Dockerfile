FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Explicitly copy models directory (ensure models are included)
# Copy models directory with all files
COPY models/ ./models/

# Verify models exist and show what was copied
RUN echo "=== Verifying models directory ===" && \
    ls -lah models/ && \
    echo "=== Checking for required model files ===" && \
    test -f models/tfidf_lr_model.pkl && \
    test -f models/tfidf_vectorizer.pkl && \
    test -f models/simple_nn_model.h5 && \
    test -f models/rnn_model.h5 && \
    test -f models/lstm_model.h5 && \
    test -f models/tokenizer.pkl && \
    test -f models/label_encoder.pkl && \
    test -f models/config.pkl && \
    echo "✅ All required model files found!" || \
    (echo "❌ ERROR: Missing required model files!" && exit 1)

# Expose port
EXPOSE 8080

# Set environment variable
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

