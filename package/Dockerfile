ARG PYTHON_VERSION=3.13
ARG UID=1000
ARG GID=1000

FROM registry.suse.com/bci/bci-micro:15.7 AS target

FROM registry.suse.com/bci/bci-base:15.7 AS builder
ARG PYTHON_VERSION
ARG UID
ARG GID
COPY --from=target / /target

# Install python on both builder and target.
RUN PY_ZYPP=$(echo ${PYTHON_VERSION} | tr -d '.') && \
    zypper --non-interactive install --no-recommends python${PY_ZYPP} python${PY_ZYPP}-pip && \
    zypper --non-interactive --installroot /target install --no-recommends python${PY_ZYPP} python${PY_ZYPP}-pip

# Create group and user 'ai-agent' (UID/GID 1000) with no home directory and no shell.
RUN echo "ai-agent:x:${GID}:" >> /target/etc/group && \
    echo "ai-agent:x:${UID}:${GID}:rancher-ai-agent-user:/nonexistent:/usr/bin/false" >> /target/etc/passwd

# Create /cert directory and grant ownership to ai-agent user.
RUN mkdir -p /target/cert && chown ${UID}:${GID} /target/cert && chmod 700 /target/cert

# Create python3 symlink.
RUN ln -sf python${PYTHON_VERSION} /target/usr/bin/python3

# Install dependencies via uv.
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --no-cache --system \
    --python /usr/bin/python${PYTHON_VERSION} \
    --target /target/usr/lib/python${PYTHON_VERSION}/site-packages \
    -r pyproject.toml

FROM scratch
ARG PYTHON_VERSION
ARG UID
ARG GID

COPY --from=builder /target /

# Runtime hardening: Ensure real-time logs, compatibility with read-only filesystems, and library discovery.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages \
    PATH="/usr/local/bin:/usr/bin:/bin"

WORKDIR /app
COPY --chown=${UID}:${GID} . .

USER ${UID}:${GID}

CMD ["/usr/bin/python3", "-m", "fastapi", "run", "app/main.py"]
