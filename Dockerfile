FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY app /app/app
COPY env.py /app/env.py
COPY models.py /app/models.py
COPY tasks.py /app/tasks.py
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
