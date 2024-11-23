FROM --platform=linux/amd64 python:3.11-slim-buster

WORKDIR /app

# Add environment variables
ENV FLASK_PORT=5000

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]