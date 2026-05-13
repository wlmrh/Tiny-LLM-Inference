set(TINYLLM_TEST_UNIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/unit")
set(TINYLLM_TEST_INTEGRATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/integration")
set(TINYLLM_TOOLS_DIR "${CMAKE_SOURCE_DIR}/tools")
set(TINYLLM_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts")

include(FetchContent)
include(GoogleTest)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

function(tinyllm_add_gtest target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE tiny_llm GTest::gtest_main)
    target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/include")
    gtest_discover_tests(${target})
endfunction()

tinyllm_add_gtest(test_parallel_config
    "${TINYLLM_TEST_UNIT_DIR}/test_parallel_config.cpp")

tinyllm_add_gtest(test_runtime_device_config
    "${TINYLLM_TEST_UNIT_DIR}/test_runtime_device_config.cpp")

tinyllm_add_gtest(test_weight_map
    "${TINYLLM_TEST_UNIT_DIR}/test_weight_map.cpp")

tinyllm_add_gtest(test_linear_module
    "${TINYLLM_TEST_UNIT_DIR}/test_linear_module.cpp")

tinyllm_add_gtest(test_rmsnorm_module
    "${TINYLLM_TEST_UNIT_DIR}/test_rmsnorm_module.cpp")

tinyllm_add_gtest(test_llama_ops
    "${TINYLLM_TEST_UNIT_DIR}/test_llama_ops.cpp")

tinyllm_add_gtest(test_paged_attention
    "${TINYLLM_TEST_UNIT_DIR}/test_paged_attention.cpp")

if (TINYLLM_ENABLE_CUDA)
    tinyllm_add_gtest(test_paged_attention_cuda
        "${TINYLLM_TEST_UNIT_DIR}/test_paged_attention_cuda.cpp")
endif()

tinyllm_add_gtest(test_model_runner_prepared_inputs
    "${TINYLLM_TEST_UNIT_DIR}/test_model_runner_prepared_inputs.cpp")

tinyllm_add_gtest(test_scheduler
    "${TINYLLM_TEST_UNIT_DIR}/test_scheduler.cpp")

tinyllm_add_gtest(test_kv_cache_manager
    "${TINYLLM_TEST_UNIT_DIR}/test_kv_cache_manager.cpp")

tinyllm_add_gtest(test_engine_core
    "${TINYLLM_TEST_UNIT_DIR}/test_engine_core.cpp")

tinyllm_add_gtest(test_tiny_lm_runtime
    "${TINYLLM_TEST_INTEGRATION_DIR}/test_tiny_lm_runtime.cpp")
target_compile_definitions(test_tiny_lm_runtime PRIVATE TINYLLM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

tinyllm_add_gtest(test_llm_offline
    "${TINYLLM_TEST_INTEGRATION_DIR}/test_llm_offline.cpp")

add_executable(llama_logits_dump "${TINYLLM_TOOLS_DIR}/llama_logits_dump.cpp")
target_link_libraries(llama_logits_dump PRIVATE tiny_llm)
set_target_properties(llama_logits_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(llama_tensor_dump "${TINYLLM_TOOLS_DIR}/llama_tensor_dump.cpp")
target_link_libraries(llama_tensor_dump PRIVATE tiny_llm)
set_target_properties(llama_tensor_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(llama_engine_generate "${TINYLLM_TOOLS_DIR}/llama_engine_generate.cpp")
target_link_libraries(llama_engine_generate PRIVATE tiny_llm)
set_target_properties(llama_engine_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(hf_safetensor_dump "${TINYLLM_TOOLS_DIR}/hf_safetensor_dump.cpp")
target_link_libraries(hf_safetensor_dump PRIVATE tiny_llm)
set_target_properties(hf_safetensor_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(
        NAME test_llama_logits_transformers_compare
        COMMAND
            "${Python3_EXECUTABLE}"
            "${TINYLLM_SCRIPTS_DIR}/compare_llama_logits_with_transformers.py"
            --dump-binary "$<TARGET_FILE:llama_logits_dump>"
    )
    set_tests_properties(test_llama_logits_transformers_compare PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME test_llama_tensor_transformers_compare
        COMMAND
            "${Python3_EXECUTABLE}"
            "${TINYLLM_SCRIPTS_DIR}/compare_llama_tensors_with_transformers.py"
            --dump-binary "$<TARGET_FILE:llama_tensor_dump>"
    )
    set_tests_properties(test_llama_tensor_transformers_compare PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME test_llama_generation_transformers_compare
        COMMAND
            "${Python3_EXECUTABLE}"
            "${TINYLLM_SCRIPTS_DIR}/compare_llama_generation_with_transformers.py"
            --engine-binary "$<TARGET_FILE:llama_engine_generate>"
    )
    set_tests_properties(test_llama_generation_transformers_compare PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME test_llama_generation_cpu_smoke
        COMMAND
            "${Python3_EXECUTABLE}"
            "${TINYLLM_SCRIPTS_DIR}/run_llama_generation_smoke.py"
            --engine-binary "$<TARGET_FILE:llama_engine_generate>"
            --device cpu
            --check-known-ids
    )
    set_tests_properties(test_llama_generation_cpu_smoke PROPERTIES SKIP_RETURN_CODE 77)

    if (TINYLLM_ENABLE_CUDA)
        add_test(
            NAME test_llama_generation_cuda_smoke
            COMMAND
                "${Python3_EXECUTABLE}"
                "${TINYLLM_SCRIPTS_DIR}/run_llama_generation_smoke.py"
                --engine-binary "$<TARGET_FILE:llama_engine_generate>"
                --device cuda:0
                --check-known-ids
        )
        set_tests_properties(test_llama_generation_cuda_smoke PROPERTIES SKIP_RETURN_CODE 77)
    endif()
else()
    message(STATUS "Python3 interpreter not found; skipping Transformers comparison test registration.")
endif()
