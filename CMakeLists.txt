cmake_minimum_required(VERSION 3.18)
project(TinyLLMInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYLLM_ENABLE_CUDA "Enable CUDA backend" OFF)

find_package(Torch QUIET)
if (NOT Torch_FOUND)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
            RESULT_VARIABLE _torch_prefix_result
            OUTPUT_VARIABLE _torch_cmake_prefix
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (_torch_prefix_result EQUAL 0 AND _torch_cmake_prefix)
            list(PREPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
            find_package(Torch QUIET)
        endif()
    endif()
endif()

if (NOT Torch_FOUND AND DEFINED ENV{HOME})
    set(_torch_home "$ENV{HOME}")
    set(_torch_home_default "${_torch_home}/libtorch")

    if (EXISTS "${_torch_home_default}/share/cmake/Torch/TorchConfig.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${_torch_home_default}")
        find_package(Torch QUIET)
    endif()

    if (NOT Torch_FOUND)
        file(GLOB _torch_home_candidates "${_torch_home}/libtorch*")
        foreach(_torch_candidate IN LISTS _torch_home_candidates)
            if (EXISTS "${_torch_candidate}/share/cmake/Torch/TorchConfig.cmake")
                list(PREPEND CMAKE_PREFIX_PATH "${_torch_candidate}")
                find_package(Torch QUIET)
                if (Torch_FOUND)
                    break()
                endif()
            endif()
        endforeach()
    endif()
endif()

if (NOT Torch_FOUND)
    message(FATAL_ERROR "Torch not found. Set Torch_DIR/CMAKE_PREFIX_PATH to libtorch, or install Python torch.")
endif()

if (TORCH_CXX_FLAGS)
    string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
endif()

if (TINYLLM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_library(tiny_llm_core STATIC
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
    target_link_libraries(tiny_llm_core PUBLIC CUDA::cudart ${TORCH_LIBRARIES})
else()
    target_compile_definitions(tiny_llm_core PUBLIC TINYLLM_ENABLE_CUDA=0)
    target_link_libraries(tiny_llm_core PUBLIC ${TORCH_LIBRARIES})
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
target_link_libraries(tiny_llm_core PUBLIC tiny_llm_models)

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