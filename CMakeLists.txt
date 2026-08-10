cmake_minimum_required(VERSION 3.20)
project(tensor_integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_METAL ON CACHE BOOL "" FORCE)
add_subdirectory(include/ggml)

add_library(tllm STATIC
    src/gguf/loader.cpp
    src/gguf/metadata.cpp
    src/model/config.cpp
    src/model/model.cpp
    src/tokenizer/tokenizer.cpp
    src/ops/norm.cpp
    src/ops/rope.cpp
    src/ops/trans.cpp
    src/ops/ffn.cpp
    src/ops/attn.cpp
    src/ops/embed.cpp
    src/runtime/compute.cpp
    src/runtime/backend.cpp
    src/runtime/kv_cache.cpp
)

target_include_directories(tllm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ggml/include
)

target_link_libraries(tllm PUBLIC ggml)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE tllm)
