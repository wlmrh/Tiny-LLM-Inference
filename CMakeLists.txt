cmake_minimum_required(VERSION 3.18)
project(TinyLLMInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 是否启用 CUDA 编译
option(TINYLLM_ENABLE_CUDA "Enable CUDA backend" OFF)

# ==============================================================================
# 1. 核心依赖检测 (注意顺序：必须先找 Torch 注入 ABI，再编译第三方源码)
# ==============================================================================

# -------- 第一步：寻找 PyTorch --------
find_package(Torch QUIET)
if (NOT Torch_FOUND)
    # 尝试从 Python 虚拟环境提取 Torch CMake 路径
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

# 备用方案：在 HOME 目录寻找
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

# 🔥 核心防御：将 Torch 的 ABI 标志 (-D_GLIBCXX_USE_CXX11_ABI=0) 注入全局
if (TORCH_CXX_FLAGS)
    string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
endif()

# -------- 第二步：配置 CUDA 环境 --------
# Keep this before FetchContent dependencies so CUDA configure failures are fast
# and do not trigger unnecessary third-party fetch/build work.
if (TINYLLM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# -------- 第三步：编译 tokenizers_cpp (此时它会自动继承上方注入的旧版 ABI) --------
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
find_program(TINYLLM_CARGO_EXECUTABLE cargo)
if (NOT TINYLLM_CARGO_EXECUTABLE)
    message(FATAL_ERROR "tokenizers-cpp requires Rust cargo. Please install Rust first: https://rustup.rs/")
endif()

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG main
)
FetchContent_MakeAvailable(tokenizers_cpp)

# ==============================================================================
# 2. 核心引擎：构建单一全局静态库 (tiny_llm)
# 将所有模块打包在一起，从物理层面彻底消灭交叉依赖和链接顺序问题！
# ==============================================================================
add_library(tiny_llm STATIC
    # --- Core 模块 ---
    src/core/execution_context.cpp
    src/core/allocator_common.cpp
    src/runtime/tokenizer.cpp
    src/runtime/request.cpp
    src/runtime/parallel_config.cpp
    src/runtime/scheduler.cpp
    src/runtime/execution_context.cpp
    src/runtime/sampler.cpp
    src/runtime/processors.cpp
    src/runtime/model_runner.cpp
    src/runtime/kv_cache.cpp
    src/runtime/engine_core.cpp
    src/runtime/engine.cpp
    
    # --- Operators 模块 ---
    src/operators/rmsnorm/rmsnorm.cpp
    src/operators/matmul/matmul.cpp
    src/operators/llama_ops.cpp
    src/operators/paged_attention/paged_attention.cpp
    src/operators/paged_attention/paged_attention_cpu.cpp
    src/operators/paged_attention/paged_attention_torch.cpp
    
    # --- Models 模块 ---
    src/models/hf_llama_config_loader.cpp
    src/models/hf_safetensors_loader.cpp
    src/models/llama_layer.cpp
    src/models/llama_model.cpp
    src/models/llama_weight_map.cpp
    src/models/mini_llama.cpp
    src/models/modules/embedding.cpp
    src/models/modules/linear.cpp
    src/models/modules/rmsnorm.cpp
    src/models/modules/rotary_embedding.cpp
    src/models/tiny_lm.cpp
)

target_include_directories(tiny_llm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# 绑定后端实现与链接库
if (TINYLLM_ENABLE_CUDA)
    target_sources(tiny_llm PRIVATE
        src/core/cuda/cuda_allocator.cpp
        src/operators/rmsnorm/rmsnorm.cu
        src/operators/matmul/gemm_kernels.cu
        src/operators/paged_attention/paged_attention_kernels.cu
    )
    target_compile_definitions(tiny_llm PUBLIC TINYLLM_ENABLE_CUDA=1)
    target_link_libraries(tiny_llm PUBLIC CUDA::cudart CUDA::cublas ${TORCH_LIBRARIES} tokenizers_cpp)
else()
    target_sources(tiny_llm PRIVATE src/core/cpu/cpu_allocator.cpp)
    target_compile_definitions(tiny_llm PUBLIC TINYLLM_ENABLE_CUDA=0)
    target_link_libraries(tiny_llm PUBLIC ${TORCH_LIBRARIES} tokenizers_cpp)
endif()

# ==============================================================================
# 3. 欺骗测试文件的魔法：ALIAS 别名库
# 完美向下兼容你的 tests 目录，无需修改其他 CMakeLists.txt
# ==============================================================================
add_library(tiny_llm_core ALIAS tiny_llm)
add_library(tiny_llm_models ALIAS tiny_llm)
add_library(tiny_llm_operators ALIAS tiny_llm)

# ==============================================================================
# 4. 编译最终的可执行文件
# ==============================================================================
add_executable(llama_inference examples/llama_inference.cpp)
target_link_libraries(llama_inference PRIVATE tiny_llm)

add_executable(tiny_lm_inference examples/tiny_lm_inference.cpp)
target_link_libraries(tiny_lm_inference PRIVATE tiny_llm)

include(CTest)
add_subdirectory(tests)
