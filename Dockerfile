FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000

CMD ["uvicorn", "energivanu.api:app", "--host", "0.0.0.0", "--port", "8000"]
