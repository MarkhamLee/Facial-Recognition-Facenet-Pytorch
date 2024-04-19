FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.12.2-slim

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

COPY *.py ./

EXPOSE 6000

ENTRYPOINT ["gunicorn" , "--bind", "0.0.0.0:6000",  "wsgi:app"]