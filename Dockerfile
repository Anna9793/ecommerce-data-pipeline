FROM python:3.9.6 AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unit tests inside the container
FROM builder AS testrunner
ENV USE_BIGQUERY=false
RUN PYTHONPATH=. pytest && touch .test_passed

# Final production runner
FROM builder AS final
COPY --from=testrunner /app/.test_passed /app/.test_passed

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]