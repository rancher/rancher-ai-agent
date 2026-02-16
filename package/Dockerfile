FROM registry.suse.com/bci/python:3.13

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv pip install --system -r pyproject.toml

# Create non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["fastapi", "run", "app/main.py"]