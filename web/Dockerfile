# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# gomoku-web: the static React build, served by Caddy.
# Build context: web/
#
#   docker build -t gomoku-web web
# -----------------------------------------------------------------------------
ARG NODE_VERSION=24
ARG CADDY_VERSION=2.11

FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca; fi; \
    pnpm install --frozen-lockfile
COPY . .
ARG VERSION=""
ARG REVISION=unknown
RUN APP_VERSION="${VERSION}" GIT_SHA="${REVISION}" sh -c '[ -n "$APP_VERSION" ] || unset APP_VERSION; pnpm run build'

FROM caddy:${CADDY_VERSION}-alpine AS runtime
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="gomoku-web" \
      org.opencontainers.image.source="https://github.com/dominik-skorka/gomoku" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/healthz"]
