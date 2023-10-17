FROM python:3.9.10-slim-buster

COPY requirements.txt ./requirements.txt
RUN pip3 install -r /requirements.txt

COPY wsgi.py .
COPY server.py .
COPY photo_inferencing.py .
COPY score_service.py .

EXPOSE 6000

CMD ["gunicorn" , "--bind", "0.0.0.0:6000",  "wsgi:app"]