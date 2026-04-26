add_executable(test_tiny_lm_runtime test_tiny_lm_runtime.cpp)
target_link_libraries(test_tiny_lm_runtime PRIVATE tiny_llm_models)
target_compile_definitions(test_tiny_lm_runtime PRIVATE TINYLLM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME test_tiny_lm_runtime COMMAND test_tiny_lm_runtime)

add_executable(test_llama_phase1 test_llama_phase1.cpp)
target_link_libraries(test_llama_phase1 PRIVATE tiny_llm_models)
add_test(NAME test_llama_phase1 COMMAND test_llama_phase1)

add_executable(test_llama_phase2 test_llama_phase2.cpp)
target_link_libraries(test_llama_phase2 PRIVATE tiny_llm_models)
add_test(NAME test_llama_phase2 COMMAND test_llama_phase2)

add_executable(test_llama_phase3 test_llama_phase3.cpp)
target_link_libraries(test_llama_phase3 PRIVATE tiny_llm_models)
add_test(NAME test_llama_phase3 COMMAND test_llama_phase3)

add_executable(test_llama_phase4 test_llama_phase4.cpp)
target_link_libraries(test_llama_phase4 PRIVATE tiny_llm_models)
add_test(NAME test_llama_phase4 COMMAND test_llama_phase4)

add_executable(llama_logits_dump llama_logits_dump.cpp)
target_link_libraries(llama_logits_dump PRIVATE tiny_llm_models)

find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(
        NAME test_llama_logits_transformers_compare
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/compare_llama_logits_with_transformers.py"
            --dump-binary "$<TARGET_FILE:llama_logits_dump>"
    )
    set_tests_properties(test_llama_logits_transformers_compare PROPERTIES SKIP_RETURN_CODE 77)
else()
    message(STATUS "Python3 interpreter not found; skipping Transformers logits comparison test registration.")
endif()

add_executable(hf_safetensor_dump hf_safetensor_dump.cpp)
target_link_libraries(hf_safetensor_dump PRIVATE tiny_llm_models)
