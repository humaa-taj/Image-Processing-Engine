# Frontend: build the React app with Node, then serve the static files with a small nginx image.
# Build context = frontend/ (see .dockerignore). No models, no data.

# ---- stage 1: build ----
FROM node:22.19.0-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# Real API only (mock mode off). The Experiments link points to the MLflow UI on the host.
ARG VITE_MLFLOW_URL=http://127.0.0.1:5000
ENV VITE_MOCK=0 VITE_MLFLOW_URL=$VITE_MLFLOW_URL
RUN npm run build

# ---- stage 2: serve ----
FROM nginx:1.29.1-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1
