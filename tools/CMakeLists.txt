set(TINYLLM_TOOL_TARGETS
    hf_safetensor_dump
    llama_engine_generate
    llama_logits_dump
    llama_tensor_dump)

foreach(target IN LISTS TINYLLM_TOOL_TARGETS)
    add_executable(${target} "${target}.cpp")
    target_link_libraries(${target} PRIVATE TinyLLM::tiny_llm)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
endforeach()
