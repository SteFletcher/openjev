FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OPENJEV_HOST=0.0.0.0 OPENJEV_PORT=8080
WORKDIR /app
COPY pyproject.toml README.md ./
COPY openjev ./openjev
RUN pip install --no-cache-dir .
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["python", "-m", "openjev"]
