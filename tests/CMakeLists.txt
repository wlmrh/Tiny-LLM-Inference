# tests/CMakeLists.txt

add_executable(dummy_test tests_dummy.cpp)

# Link against the portable CPU-only library (available on macOS).
target_link_libraries(dummy_test PRIVATE tiny_llm_logic)

add_test(NAME dummy_test COMMAND dummy_test)