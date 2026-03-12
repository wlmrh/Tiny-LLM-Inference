cmake_minimum_required(VERSION 3.18)
project(TinyLLMInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYLLM_ENABLE_CUDA "Build CUDA kernels and GPU operators" ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# -----------------------
# Logic library (CPU-only, portable)
# -----------------------
add_library(tiny_llm_logic
    src/core/tensor.cpp
    src/core/execution_context.cpp
    # Keep this target CUDA-free so it can build on macOS without NVCC.
)

target_compile_definitions(tiny_llm_logic PUBLIC
    $<$<BOOL:${TINYLLM_ENABLE_CUDA}>:TINYLLM_ENABLE_CUDA=1>
    $<$<NOT:$<BOOL:${TINYLLM_ENABLE_CUDA}>>:TINYLLM_ENABLE_CUDA=0>
)

# -----------------------
# CUDA library (only on GPU machines)
# -----------------------
if (TINYLLM_ENABLE_CUDA)
    enable_language(CUDA)

    add_library(kernels
        kernels/vector_add.cu
    )
    set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_library(tiny_llm_cuda
        src/core/allocator.cpp
        src/operators/ops.cpp
        src/models/llama_layer.cpp
    )

    target_compile_definitions(tiny_llm_cuda PUBLIC TINYLLM_ENABLE_CUDA=1)
    target_link_libraries(tiny_llm_cuda PUBLIC tiny_llm_logic kernels cuda)

    # -----------------------
    # Executable (GPU demo)
    # -----------------------
    add_executable(main_app main.cpp)
    target_link_libraries(main_app PRIVATE tiny_llm_logic tiny_llm_cuda)
endif()

# -----------------------
# Tests (CPU-only)
# -----------------------
enable_testing()
add_subdirectory(tests)