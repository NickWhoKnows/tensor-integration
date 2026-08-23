cmake_minimum_required(VERSION 3.20)
project(tensor_integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_METAL OFF CACHE BOOL "" FORCE)
add_subdirectory(include/ggml)

add_library(tllm STATIC
    src/cli.cpp
    src/gguf/loader.cpp
    src/gguf/metadata.cpp
    src/model/config.cpp
    src/model/model.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/chat_template.cpp
    src/ops/layer.cpp
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
