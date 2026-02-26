FROM python:3.10.11-slim-bookworm

WORKDIR /app

COPY ./src/requirements.txt /app/src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/src/requirements.txt

COPY ./src /app/src
COPY ./resources /app/resources

ENV PYTHONPATH=/app

WORKDIR /app/src
RUN python main.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]