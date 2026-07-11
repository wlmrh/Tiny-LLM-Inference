FROM ubuntu:22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git pkg-config python3 python3-pip libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal \
    && rm /tmp/rustup-init.sh
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY . .
RUN cmake --preset ci-cpu \
    && cmake --build --preset ci-cpu --parallel 2 \
    && ctest --preset ci-cpu

FROM ubuntu:22.04
COPY --from=build /src/build/ci-cpu/offline_llm /usr/local/bin/tinyllm-offline
ENTRYPOINT ["/usr/local/bin/tinyllm-offline"]
