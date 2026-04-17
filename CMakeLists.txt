cmake_minimum_required(VERSION 3.18)
project(TinyLLMInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYLLM_ENABLE_CUDA "Enable CUDA backend" OFF)

if (TINYLLM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_library(tiny_llm_core STATIC
    src/core/tensor.cpp
    src/core/execution_context.cpp
    src/core/allocator_common.cpp
    src/runtime/tokenizer.cpp
    src/runtime/request.cpp
    src/runtime/scheduler.cpp
    src/runtime/execution_context.cpp
    src/runtime/processors.cpp
    src/runtime/executor.cpp
    src/runtime/kv_cache.cpp
    src/runtime/engine_core.cpp
    src/runtime/engine.cpp
)

if (TINYLLM_ENABLE_CUDA)
    target_sources(tiny_llm_core PRIVATE
        src/core/cuda/cuda_allocator.cpp
    )
else()
    target_sources(tiny_llm_core PRIVATE
        src/core/cpu/cpu_allocator.cpp
    )
endif()

target_include_directories(tiny_llm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if (TINYLLM_ENABLE_CUDA)
    target_compile_definitions(tiny_llm_core PUBLIC TINYLLM_ENABLE_CUDA=1)
    target_link_libraries(tiny_llm_core PUBLIC CUDA::cudart)
else()
    target_compile_definitions(tiny_llm_core PUBLIC TINYLLM_ENABLE_CUDA=0)
endif()

add_library(tiny_llm_operators STATIC
    src/operators/rmsnorm/rmsnorm.cpp
    src/operators/matmul/matmul.cpp
    src/operators/paged_attention/paged_attention.cpp
)

target_link_libraries(tiny_llm_operators PUBLIC tiny_llm_core)

if (TINYLLM_ENABLE_CUDA)
    target_sources(tiny_llm_operators PRIVATE
        src/operators/rmsnorm/rmsnorm.cu
        src/operators/matmul/gemm_kernels.cu
        src/operators/paged_attention/paged_attention_kernels.cu
    )
    target_link_libraries(tiny_llm_operators PUBLIC CUDA::cublas)
endif()

add_library(tiny_llm_models STATIC
    src/models/llama_layer.cpp
    src/models/tiny_lm.cpp
)

target_link_libraries(tiny_llm_models PUBLIC tiny_llm_operators tiny_llm_core)

add_executable(llama_inference
    examples/llama_inference.cpp
)

target_link_libraries(llama_inference PRIVATE tiny_llm_models)

add_executable(tiny_lm_inference
    examples/tiny_lm_inference.cpp
)

target_link_libraries(tiny_lm_inference PRIVATE tiny_llm_models)

include(CTest)
add_subdirectory(tests)