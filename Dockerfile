FROM python:3.11-slim AS builder

# Install uv from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies using uv for high-performance builds
COPY requirements.txt pyproject.toml ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

# Run unit tests inside the container during build
FROM builder AS testrunner
ENV USE_BIGQUERY=false
RUN PYTHONPATH=. pytest && touch .test_passed

# Final production runner
FROM builder AS final
COPY --from=testrunner /app/.test_passed /app/.test_passed

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]