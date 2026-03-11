# tests/CMakeLists.txt

add_executable(dummy_test tests_dummy.cpp)
target_link_libraries(dummy_test PRIVATE tiny_llm_core)
add_test(NAME dummy_test COMMAND dummy_test)
