# syntax=docker/dockerfile:1

# ============ STAGE 1: Build the application in CUDA image =============
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Go (adjust version as needed)
ENV GOLANG_VERSION=1.20.3
RUN wget https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-amd64.tar.gz && \
    rm go${GOLANG_VERSION}.linux-amd64.tar.gz

# Set up Go environment
ENV GOPATH=/go
ENV PATH="/usr/local/go/bin:${PATH}:${GOPATH}/bin"

WORKDIR /app

# Copy local source into the container
COPY . /app

# Build the PTX and the Go binary
RUN make clean && make all

# ============ STAGE 2: Minimal Distroless image =============
FROM gcr.io/distroless/base-debian11

# We copy only the artifacts needed for runtime:
WORKDIR /app

COPY --from=builder /app/cuda-go     /app/cuda-go
COPY --from=builder /app/matmul.ptx  /app/matmul.ptx

# Distroless requires specifying an entrypoint (no shell by default)
ENTRYPOINT ["/app/cuda-go"]
