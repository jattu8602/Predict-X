FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/index.html dashboard/vite.config.js ./
COPY dashboard/src ./src
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY src ./src
COPY deploy/fetch_fd001.py /tmp/fetch_fd001.py
RUN python /tmp/fetch_fd001.py \
    && python -m src.train_model \
    && rm data/raw/train_FD001.txt \
    && chmod -R a+rX data/raw models
COPY --from=frontend /build/dist ./dashboard/dist
USER 10001:10001
EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn src.web:app --host 0.0.0.0 --port ${PORT:-10000}"]
