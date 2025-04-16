FROM python:3.10-slim

WORKDIR /app

# 🧠 Add curl here
RUN apt-get update && apt-get install -y curl gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
