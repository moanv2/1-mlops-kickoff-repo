FROM python:3.11-slim

WORKDIR /app

# Install pinned production dependencies (derived from conda-lock.yml)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the serving image needs
COPY config.yaml .
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
