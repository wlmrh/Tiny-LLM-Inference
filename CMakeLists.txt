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
find_package(Python3 COMPONENTS Interpreter QUIET)

function(tinyllm_torch_prefix_from_cmake_path cmake_path out_var)
    set(_torch_prefix "")
    if (cmake_path AND EXISTS "${cmake_path}/TorchConfig.cmake")
        get_filename_component(_torch_prefix "${cmake_path}/../../.." ABSOLUTE)
    elseif (cmake_path AND EXISTS "${cmake_path}/Torch/TorchConfig.cmake")
        get_filename_component(_torch_prefix "${cmake_path}/../.." ABSOLUTE)
    elseif (cmake_path AND EXISTS "${cmake_path}/share/cmake/Torch/TorchConfig.cmake")
        get_filename_component(_torch_prefix "${cmake_path}" ABSOLUTE)
    endif()
    set(${out_var} "${_torch_prefix}" PARENT_SCOPE)
endfunction()

function(tinyllm_torch_prefix_from_torch_dir out_var)
    set(_torch_prefix "")
    if (Torch_DIR AND EXISTS "${Torch_DIR}/TorchConfig.cmake")
        tinyllm_torch_prefix_from_cmake_path("${Torch_DIR}" _torch_prefix)
    endif()
    set(${out_var} "${_torch_prefix}" PARENT_SCOPE)
endfunction()

function(tinyllm_torch_prefix_from_cmake_prefix_path out_var)
    set(_torch_prefix "")
    foreach(_cmake_prefix IN LISTS CMAKE_PREFIX_PATH)
        tinyllm_torch_prefix_from_cmake_path("${_cmake_prefix}" _candidate_prefix)
        if (_candidate_prefix)
            set(_torch_prefix "${_candidate_prefix}")
            break()
        endif()
    endforeach()
    set(${out_var} "${_torch_prefix}" PARENT_SCOPE)
endfunction()

function(tinyllm_python_torch_prefix out_var)
    set(_torch_prefix "")
    if (Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
            RESULT_VARIABLE _torch_prefix_result
            OUTPUT_VARIABLE _torch_cmake_prefix
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (_torch_prefix_result EQUAL 0 AND _torch_cmake_prefix)
            tinyllm_torch_prefix_from_cmake_path("${_torch_cmake_prefix}" _torch_prefix)
        endif()
    endif()
    set(${out_var} "${_torch_prefix}" PARENT_SCOPE)
endfunction()

macro(tinyllm_clear_torch_package_cache)
    foreach(_torch_cache_var IN ITEMS
        TORCH_LIBRARY
        TORCH_CPU_LIBRARY
        TORCH_CUDA_LIBRARY
        c10_LIBRARY
        C10_LIBRARY
        C10_CUDA_LIBRARY
        kineto_LIBRARY
        Caffe2_DIR
        CAFFE2_NVRTC_LIBRARY
    )
        unset(${_torch_cache_var})
        unset(${_torch_cache_var} CACHE)
    endforeach()
    unset(Torch_FOUND)
endmacro()

function(tinyllm_prepare_torch_cuda_compiler torch_prefix)
    if (CMAKE_CUDA_COMPILER)
        return()
    endif()

    set(_nvcc_candidates "")
    if (DEFINED ENV{CUDACXX})
        list(APPEND _nvcc_candidates "$ENV{CUDACXX}")
    endif()
    foreach(_cuda_root IN ITEMS
        "${CUDAToolkit_ROOT}"
        "$ENV{CUDAToolkit_ROOT}"
        "$ENV{CUDA_HOME}"
        "$ENV{CUDA_PATH}"
        "/usr/local/cuda")
        if (_cuda_root)
            list(APPEND _nvcc_candidates "${_cuda_root}/bin/nvcc")
        endif()
    endforeach()

    foreach(_nvcc IN LISTS _nvcc_candidates)
        if (_nvcc AND EXISTS "${_nvcc}")
            set(CMAKE_CUDA_COMPILER "${_nvcc}" CACHE FILEPATH "CUDA compiler used when a CUDA-enabled Torch package configures Caffe2." FORCE)
            return()
        endif()
    endforeach()

    if (torch_prefix AND EXISTS "${torch_prefix}/lib/libtorch_cuda.so")
        message(FATAL_ERROR
            "The Torch package at ${torch_prefix} includes CUDA libraries, so its CMake package may need a CUDA compiler even when TINYLLM_ENABLE_CUDA=OFF. "
            "Set CUDACXX or CMAKE_CUDA_COMPILER to nvcc, install CUDA toolkit, or use a CPU-only libtorch package.")
    endif()
endfunction()

function(tinyllm_prepare_missing_torch_optional_libraries torch_prefix)
    if (NOT torch_prefix)
        return()
    endif()

    get_filename_component(_torch_prefix "${torch_prefix}" ABSOLUTE)
    if (EXISTS "${_torch_prefix}/lib/libkineto.a")
        return()
    endif()

    if (CMAKE_AR AND (EXISTS "${_torch_prefix}/lib/libtorch${CMAKE_SHARED_LIBRARY_SUFFIX}" OR EXISTS "${_torch_prefix}/lib/libtorch.dylib" OR EXISTS "${_torch_prefix}/lib/libtorch.so"))
        set(_kineto_stub_dir "${CMAKE_BINARY_DIR}/tinyllm_torch_optional_stubs")
        set(_kineto_stub_source "${_kineto_stub_dir}/kineto_stub.cpp")
        set(_kineto_stub_object "${_kineto_stub_dir}/kineto_stub.o")
        set(_kineto_stub "${_kineto_stub_dir}/libkineto.a")
        file(MAKE_DIRECTORY "${_kineto_stub_dir}")
        if (NOT EXISTS "${_kineto_stub}")
            file(WRITE "${_kineto_stub_source}" "int tinyllm_kineto_stub = 0;\n")
            execute_process(
                COMMAND "${CMAKE_CXX_COMPILER}" -c "${_kineto_stub_source}" -o "${_kineto_stub_object}"
                RESULT_VARIABLE _kineto_stub_compile_result
                ERROR_QUIET
            )
            if (_kineto_stub_compile_result EQUAL 0)
                execute_process(COMMAND "${CMAKE_AR}" qc "${_kineto_stub}" "${_kineto_stub_object}" RESULT_VARIABLE _kineto_stub_result)
                if (_kineto_stub_result EQUAL 0 AND CMAKE_RANLIB)
                    execute_process(COMMAND "${CMAKE_RANLIB}" "${_kineto_stub}" ERROR_QUIET)
                endif()
            endif()
        endif()
        if (EXISTS "${_kineto_stub}")
            set(kineto_LIBRARY "${_kineto_stub}" CACHE FILEPATH "Empty optional kineto archive for Torch packages that do not ship libkineto.a." FORCE)
        endif()
    endif()
endfunction()

function(tinyllm_check_torch_prefix_consistency torch_prefix)
    get_filename_component(_torch_prefix "${torch_prefix}" ABSOLUTE)
    set(_mismatches "")
    foreach(_torch_path_var IN ITEMS
        Torch_DIR
        Caffe2_DIR
        TORCH_LIBRARY
        c10_LIBRARY
        C10_CUDA_LIBRARY)
        if (DEFINED ${_torch_path_var} AND ${_torch_path_var})
            set(_torch_path_value "${${_torch_path_var}}")
            if (_torch_path_value MATCHES "-NOTFOUND$")
                continue()
            endif()
            get_filename_component(_torch_path_abs "${_torch_path_value}" ABSOLUTE)
            string(FIND "${_torch_path_abs}" "${_torch_prefix}/" _prefix_pos)
            if (NOT _prefix_pos EQUAL 0 AND NOT _torch_path_abs STREQUAL "${_torch_prefix}")
                list(APPEND _mismatches "${_torch_path_var}=${_torch_path_value}")
            endif()
        endif()
    endforeach()

    if (_mismatches)
        string(REPLACE ";" "\n  " _mismatch_text "${_mismatches}")
        message(FATAL_ERROR
            "Torch dependency pollution detected. Canonical Torch prefix is ${_torch_prefix}, but these cached paths point elsewhere:\n  ${_mismatch_text}\n"
            "Reconfigure with a single -DTorch_DIR=<torch-prefix>/share/cmake/Torch, or remove stale CMakeCache.txt/CMakeFiles entries.")
    endif()
endfunction()

macro(tinyllm_find_torch_at_prefix torch_prefix)
    if (NOT "${torch_prefix}" STREQUAL "")
        get_filename_component(_tinyllm_torch_prefix "${torch_prefix}" ABSOLUTE)
        set(TINYLLM_TORCH_PREFIX "${_tinyllm_torch_prefix}" CACHE PATH "Canonical Torch installation prefix used by TinyLLM." FORCE)
        tinyllm_clear_torch_package_cache()
        set(Torch_DIR "${TINYLLM_TORCH_PREFIX}/share/cmake/Torch" CACHE PATH "Torch CMake package directory." FORCE)
        set(Caffe2_DIR "${TINYLLM_TORCH_PREFIX}/share/cmake/Caffe2" CACHE PATH "Caffe2 CMake package directory matching Torch_DIR." FORCE)
        tinyllm_prepare_missing_torch_optional_libraries("${TINYLLM_TORCH_PREFIX}")
        tinyllm_prepare_torch_cuda_compiler("${TINYLLM_TORCH_PREFIX}")
        find_package(Torch QUIET CONFIG)
        if (Torch_FOUND)
            tinyllm_check_torch_prefix_consistency("${TINYLLM_TORCH_PREFIX}")
            message(STATUS "Using Torch from ${TINYLLM_TORCH_PREFIX}")
        endif()
    endif()
endmacro()

macro(tinyllm_find_torch_from_candidates)
    set(_tinyllm_torch_candidate "")

    # 1. Respect an explicit Torch_DIR first. This also repairs stale Caffe2/c10 cache entries.
    tinyllm_torch_prefix_from_torch_dir(_tinyllm_torch_candidate)
    if (_tinyllm_torch_candidate)
        tinyllm_find_torch_at_prefix("${_tinyllm_torch_candidate}")
    endif()

    # 2. Then honor explicit CMAKE_PREFIX_PATH entries without permanently appending Python paths to it.
    if (NOT Torch_FOUND)
        tinyllm_torch_prefix_from_cmake_prefix_path(_tinyllm_torch_candidate)
        if (_tinyllm_torch_candidate)
            tinyllm_find_torch_at_prefix("${_tinyllm_torch_candidate}")
        endif()
    endif()

    # 3. Fall back to the active Python torch package.
    if (NOT Torch_FOUND)
        tinyllm_python_torch_prefix(_tinyllm_torch_candidate)
        if (_tinyllm_torch_candidate)
            tinyllm_find_torch_at_prefix("${_tinyllm_torch_candidate}")
        endif()
    endif()

    # 4. Finally search common libtorch installs under HOME.
    if (NOT Torch_FOUND AND DEFINED ENV{HOME})
        set(_torch_home "$ENV{HOME}")
        set(_torch_home_default "${_torch_home}/libtorch")
        if (EXISTS "${_torch_home_default}/share/cmake/Torch/TorchConfig.cmake")
            tinyllm_find_torch_at_prefix("${_torch_home_default}")
        endif()
        if (NOT Torch_FOUND)
            file(GLOB _torch_home_candidates "${_torch_home}/libtorch*")
            foreach(_torch_candidate IN LISTS _torch_home_candidates)
                if (EXISTS "${_torch_candidate}/share/cmake/Torch/TorchConfig.cmake")
                    tinyllm_find_torch_at_prefix("${_torch_candidate}")
                    if (Torch_FOUND)
                        break()
                    endif()
                endif()
            endforeach()
        endif()
    endif()
endmacro()

tinyllm_find_torch_from_candidates()

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
    src/runtime/parallel_config.cpp
    src/runtime/scheduler.cpp
    src/runtime/execution_context.cpp
    src/runtime/sampler.cpp
    src/runtime/profiling.cpp
    src/runtime/runtime_stats.cpp
    src/runtime/processors.cpp
    src/runtime/model_runner.cpp
    src/runtime/kv_cache.cpp
    src/runtime/engine_core.cpp
    src/runtime/engine.cpp
    src/runtime/llm.cpp
    src/runtime/generation_config.cpp
    
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
    src/models/modules/embedding.cpp
    src/models/modules/linear.cpp
    src/models/modules/rmsnorm.cpp
    src/models/modules/rotary_embedding.cpp
)

target_include_directories(tiny_llm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# 绑定后端实现与链接库
if (TINYLLM_ENABLE_CUDA)
    target_sources(tiny_llm PRIVATE
        src/core/cuda/cuda_allocator.cpp
        src/operators/rmsnorm/rmsnorm.cu
        src/operators/matmul/gemm_kernels.cu
        src/operators/llama_ops_kernels.cu
        src/operators/paged_attention/paged_attention_kernels.cu
        src/runtime/sampler_kernels.cu
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
add_executable(offline_llm examples/offline_llm.cpp)
target_link_libraries(offline_llm PRIVATE tiny_llm)

add_subdirectory(benchmark)

include(CTest)
add_subdirectory(tests)
