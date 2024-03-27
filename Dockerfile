FROM python:3.11.8-slim
WORKDIR /
COPY ./requirements.txt /llmops/requirements.txt
RUN pip install --no-cache-dir -r /llmops/requirements.txt
COPY ./app.py /app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]