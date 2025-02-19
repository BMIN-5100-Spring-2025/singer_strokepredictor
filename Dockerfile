FROM python:3.10

WORKDIR /

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY app /app

CMD ["python3.10", "/app/main.py"]