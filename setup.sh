#!/usr/bin/env bash
set -euo pipefail

preset="${1:-cpu-release}"
case "${preset}" in
    cpu-debug|cpu-release|cuda-release) ;;
    *) echo "Usage: $0 [cpu-debug|cpu-release|cuda-release]" >&2; exit 2 ;;
esac

for command in cmake python3 cargo; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing required command: ${command}" >&2
        exit 1
    fi
done

python3 -c 'import torch; print("PyTorch", torch.__version__, "CMake", torch.utils.cmake_prefix_path)'

if [[ "${preset}" == "cuda-release" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "cuda-release requires nvcc in PATH or CUDACXX." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

cmake --preset "${preset}"
cmake --build --preset "${preset}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

if [[ "${preset}" == "cuda-release" ]]; then
    ctest --preset cuda-release
else
    ctest --test-dir "build/${preset}" --output-on-failure
fi
