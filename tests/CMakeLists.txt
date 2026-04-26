set(TINYLLM_TEST_UNIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/unit")
set(TINYLLM_TEST_INTEGRATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/integration")
set(TINYLLM_TOOLS_DIR "${CMAKE_SOURCE_DIR}/tools")
set(TINYLLM_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts")

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
else()
    message(STATUS "Python3 interpreter not found; skipping Transformers comparison test registration.")
endif()
