cmake_minimum_required(VERSION 3.18)
project(TinyLLMInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYLLM_ENABLE_CUDA "Build CUDA kernels and GPU operators" ON)
option(ENABLE_FUSION "Enable fused kernels" OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

# -----------------------
# Core / logic library
# -----------------------
add_library(tiny_llm_core
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/models/llama_layer.cpp
    src/operators/ops.cpp
)
target_compile_definitions(tiny_llm_core PUBLIC
    $<$<BOOL:${TINYLLM_ENABLE_CUDA}>:TINYLLM_ENABLE_CUDA=1>
    $<$<NOT:$<BOOL:${TINYLLM_ENABLE_CUDA}>>:TINYLLM_ENABLE_CUDA=0>
)

# -----------------------
# CUDA kernels (optional)
# -----------------------
if (TINYLLM_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(kernels
        kernels/vector_add.cu
    )
    set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(tiny_llm_core PUBLIC cuda)
endif()

# -----------------------
# Executable
# -----------------------
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE tiny_llm_core)
if (TINYLLM_ENABLE_CUDA)
    target_link_libraries(main_app PRIVATE kernels cuda)
endif()

# -----------------------
# Tests
# -----------------------
enable_testing()
add_subdirectory(tests)