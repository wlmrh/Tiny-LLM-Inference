cmake_minimum_required(VERSION 3.18)
project(TinyLLMInference LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(ENABLE_FUSION "Enable fused kernels" OFF)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# core library
add_library(tiny_llm_core
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/models/llama_layer.cpp
    src/operators/ops.cpp
)

target_link_libraries(tiny_llm_core PUBLIC cuda)

# executable
add_executable(main_app
    main.cpp
)

target_link_libraries(main_app PRIVATE tiny_llm_core cuda)

# kernels as separate library (compiled with CUDA)
add_library(kernels
    kernels/vector_add.cu
)

set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(main_app PRIVATE kernels)

# enable testing
enable_testing()

# add tests (placeholder)
# add_subdirectory(tests)
