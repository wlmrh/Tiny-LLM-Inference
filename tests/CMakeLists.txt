set(TINYLLM_TEST_UNIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/unit")
set(TINYLLM_TEST_INTEGRATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/integration")
set(TINYLLM_TOOLS_DIR "${CMAKE_SOURCE_DIR}/tools")
set(TINYLLM_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts")

add_executable(test_parallel_config
    "${TINYLLM_TEST_UNIT_DIR}/test_parallel_config.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/parallel_config.cpp"
)
target_include_directories(test_parallel_config PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_parallel_config PRIVATE c10)
add_test(NAME test_parallel_config COMMAND test_parallel_config)

add_executable(test_runtime_device_config
    "${TINYLLM_TEST_UNIT_DIR}/test_runtime_device_config.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/parallel_config.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/execution_context.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/kv_cache.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/allocator_common.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/cpu/cpu_allocator.cpp"
)
target_include_directories(test_runtime_device_config PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_runtime_device_config PRIVATE c10 ${TORCH_LIBRARIES})
add_test(NAME test_runtime_device_config COMMAND test_runtime_device_config)

add_executable(test_weight_map
    "${TINYLLM_TEST_UNIT_DIR}/test_weight_map.cpp"
    "${CMAKE_SOURCE_DIR}/src/models/llama_weight_map.cpp"
    "${CMAKE_SOURCE_DIR}/src/models/hf_safetensors_loader.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/parallel_config.cpp"
)
target_include_directories(test_weight_map PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_weight_map PRIVATE c10 ${TORCH_LIBRARIES})
add_test(NAME test_weight_map COMMAND test_weight_map)

add_executable(test_linear_module
    "${TINYLLM_TEST_UNIT_DIR}/test_linear_module.cpp"
    "${CMAKE_SOURCE_DIR}/src/models/modules/linear.cpp"
    "${CMAKE_SOURCE_DIR}/src/operators/matmul/matmul.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/parallel_config.cpp"
)
target_include_directories(test_linear_module PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_linear_module PRIVATE c10 ${TORCH_LIBRARIES})
add_test(NAME test_linear_module COMMAND test_linear_module)

add_executable(test_rmsnorm_module
    "${TINYLLM_TEST_UNIT_DIR}/test_rmsnorm_module.cpp"
    "${CMAKE_SOURCE_DIR}/src/models/modules/rmsnorm.cpp"
    "${CMAKE_SOURCE_DIR}/src/operators/rmsnorm/rmsnorm.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/parallel_config.cpp"
)
target_include_directories(test_rmsnorm_module PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_rmsnorm_module PRIVATE c10 ${TORCH_LIBRARIES})
add_test(NAME test_rmsnorm_module COMMAND test_rmsnorm_module)

add_executable(test_model_blocks
    "${TINYLLM_TEST_UNIT_DIR}/test_model_blocks.cpp"
)
target_link_libraries(test_model_blocks PRIVATE tiny_llm_models)
add_test(NAME test_model_blocks COMMAND test_model_blocks)

add_executable(test_llama_ops
    "${TINYLLM_TEST_UNIT_DIR}/test_llama_ops.cpp"
    "${CMAKE_SOURCE_DIR}/src/operators/llama_ops.cpp"
)
target_include_directories(test_llama_ops PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_llama_ops PRIVATE c10 ${TORCH_LIBRARIES})
add_test(NAME test_llama_ops COMMAND test_llama_ops)

add_executable(test_paged_attention
    "${TINYLLM_TEST_UNIT_DIR}/test_paged_attention.cpp"
    "${CMAKE_SOURCE_DIR}/src/operators/paged_attention/paged_attention.cpp"
    "${CMAKE_SOURCE_DIR}/src/operators/paged_attention/paged_attention_cpu.cpp"
    "${CMAKE_SOURCE_DIR}/src/operators/paged_attention/paged_attention_torch.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/kv_cache.cpp"
    "${CMAKE_SOURCE_DIR}/src/runtime/parallel_config.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/allocator_common.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/cpu/cpu_allocator.cpp"
)
target_include_directories(test_paged_attention PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_paged_attention PRIVATE c10 ${TORCH_LIBRARIES})
add_test(NAME test_paged_attention COMMAND test_paged_attention)

if (TINYLLM_ENABLE_CUDA)
    add_executable(test_paged_attention_cuda
        "${TINYLLM_TEST_UNIT_DIR}/test_paged_attention_cuda.cpp"
    )
    target_link_libraries(test_paged_attention_cuda PRIVATE tiny_llm_operators)
    add_test(NAME test_paged_attention_cuda COMMAND test_paged_attention_cuda)
    set_tests_properties(test_paged_attention_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_executable(test_model_runner_prepared_inputs
    "${TINYLLM_TEST_UNIT_DIR}/test_model_runner_prepared_inputs.cpp"
)
target_link_libraries(test_model_runner_prepared_inputs PRIVATE tiny_llm_models)
add_test(NAME test_model_runner_prepared_inputs COMMAND test_model_runner_prepared_inputs)

add_executable(test_tiny_lm_runtime "${TINYLLM_TEST_INTEGRATION_DIR}/test_tiny_lm_runtime.cpp")
target_link_libraries(test_tiny_lm_runtime PRIVATE tiny_llm_models)
target_compile_definitions(test_tiny_lm_runtime PRIVATE TINYLLM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_tiny_lm_runtime COMMAND test_tiny_lm_runtime)
set_tests_properties(test_tiny_lm_runtime PROPERTIES SKIP_RETURN_CODE 77)

add_executable(llama_logits_dump "${TINYLLM_TOOLS_DIR}/llama_logits_dump.cpp")
target_link_libraries(llama_logits_dump PRIVATE tiny_llm_models)
set_target_properties(llama_logits_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(llama_tensor_dump "${TINYLLM_TOOLS_DIR}/llama_tensor_dump.cpp")
target_link_libraries(llama_tensor_dump PRIVATE tiny_llm_models)
set_target_properties(llama_tensor_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(llama_engine_generate "${TINYLLM_TOOLS_DIR}/llama_engine_generate.cpp")
target_link_libraries(llama_engine_generate PRIVATE tiny_llm_models)
set_target_properties(llama_engine_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(hf_safetensor_dump "${TINYLLM_TOOLS_DIR}/hf_safetensor_dump.cpp")
target_link_libraries(hf_safetensor_dump PRIVATE tiny_llm_models)
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
