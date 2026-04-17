add_executable(test_allocator test_allocator.cpp)
target_link_libraries(test_allocator PRIVATE tiny_llm_core)
add_test(NAME test_allocator COMMAND test_allocator)

add_executable(test_ops_cpu test_ops_cpu.cpp)
target_link_libraries(test_ops_cpu PRIVATE tiny_llm_operators)
add_test(NAME test_ops_cpu COMMAND test_ops_cpu)

add_executable(test_tiny_lm_runtime test_tiny_lm_runtime.cpp)
target_link_libraries(test_tiny_lm_runtime PRIVATE tiny_llm_models)
target_compile_definitions(test_tiny_lm_runtime PRIVATE TINYLLM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_tiny_lm_runtime COMMAND test_tiny_lm_runtime)

add_executable(test_runtime_processors test_runtime_processors.cpp)
target_link_libraries(test_runtime_processors PRIVATE tiny_llm_models)
target_compile_definitions(test_runtime_processors PRIVATE TINYLLM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_runtime_processors COMMAND test_runtime_processors)

add_executable(test_scheduler_fcfs test_scheduler_fcfs.cpp)
target_link_libraries(test_scheduler_fcfs PRIVATE tiny_llm_core)
add_test(NAME test_scheduler_fcfs COMMAND test_scheduler_fcfs)

if (TINYLLM_ENABLE_CUDA)
    add_executable(test_kernels test_kernels.cpp)
    target_link_libraries(test_kernels PRIVATE tiny_llm_operators)
    add_test(NAME test_kernels COMMAND test_kernels)
endif()