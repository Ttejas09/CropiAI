FROM python:3.11-slim

WORKDIR /app

COPY files/files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY files/files/ .

ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "app:app"]
