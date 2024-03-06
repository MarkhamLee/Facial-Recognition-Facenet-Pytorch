FROM python:3.11-slim as build

COPY requirements.txt ./requirements.txt
RUN pip3 install -r /requirements.txt

COPY wsgi.py . server.py photo_inferencing.py score_service.py logging_util.py ./


EXPOSE 6000

CMD ["gunicorn" , "--bind", "0.0.0.0:6000",  "wsgi:app"]