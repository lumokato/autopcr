ARG AUTOPCR_WEB_COMMIT=a486934361212aabdabd5cd14d83f4fc1cb7a9c9

# 固定前端源码版本，并叠加与本仓库后端协议配套的界面修改。
FROM node:22-bookworm-slim AS frontend-builder

ARG AUTOPCR_WEB_COMMIT

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global pnpm@11.9.0

WORKDIR /web

RUN curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 180 \
        "https://github.com/Lanly109/AutoPCR_Web/archive/${AUTOPCR_WEB_COMMIT}.tar.gz" \
        -o /tmp/autopcr-web.tar.gz \
    && tar -xzf /tmp/autopcr-web.tar.gz --strip-components=1 -C /web \
    && rm /tmp/autopcr-web.tar.gz

COPY frontend/overrides/pnpm-workspace.yaml ./pnpm-workspace.yaml
RUN pnpm install --frozen-lockfile

COPY frontend/overrides/src/ ./src/
RUN pnpm run build

# 阶段1：构建工具和依赖安装
FROM python:3.10-slim AS tools

# 声明获取 Docker 自动注入的架构参数
ARG TARGETARCH

RUN apt-get update \
    && apt-get upgrade -y \
    # 基础编译依赖（所有架构必须）
    && apt-get install -y build-essential libssl-dev pkg-config \
    # 动态判断架构：仅 ARM64 安装 Rust
    && if [ "$TARGETARCH" = "arm64" ]; then \
        echo "===== Installing Rust for ARM64 =====" \
        && apt-get install -y curl \
        && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        && . "$HOME/.cargo/env" \
        && rustup update stable \
        && rustup default stable; \
    fi \
    # 统一清理缓存
    && apt-get purge -y --auto-remove curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# 阶段2：安装 Python 依赖和字体
FROM tools AS builder

# 确保 Cargo 在 PATH 中
ENV PATH="/root/.cargo/bin:${PATH}"

# 安装依赖
COPY ./requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# 阶段3：最终镜像
FROM python:3.10-slim

ARG AUTOPCR_WEB_COMMIT

ENV PYTHONIOENCODING=utf-8

LABEL org.opencontainers.image.autopcr-web-commit="$AUTOPCR_WEB_COMMIT" \
      org.opencontainers.image.autopcr-web-overlay="execution-mode-20260917"

# 设置时区
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# 仅复制依赖
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 复制项目代码
COPY . .

# 使用已锁定源码构建的前端，避免未发布到 Release 的功能在镜像中丢失。
COPY --from=frontend-builder /web/dist ./autopcr/http_server/ClientApp
RUN printf 'source-%s+autopcr-overlay\n' "$AUTOPCR_WEB_COMMIT" \
        > ./autopcr/http_server/client_version \
    && rm -rf ./frontend

EXPOSE 13200

CMD ["python3", "_httpserver_test.py"]
