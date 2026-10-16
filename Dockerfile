# One container: FastAPI backend + built React viewer + MCP server entry point.
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data/matches \
    WEB_DIST=/app/web/dist
COPY pyproject.toml README.md ./
COPY sim ./sim
COPY analytics ./analytics
COPY agents ./agents
COPY mcp_server ./mcp_server
COPY api ./api
RUN pip install --no-cache-dir ".[azure]"
COPY --from=web /web/dist ./web/dist
RUN useradd --create-home app && mkdir -p /data/matches && chown -R app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
