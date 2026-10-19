FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY eval ./eval
COPY scripts ./scripts
ENV PYTHONPATH=/app/src
# Mock evaluation needs no key: docker run --rm --entrypoint python <image> -m meeting_summarizer.evaluate --mock
ENTRYPOINT ["python", "-m", "meeting_summarizer"]
