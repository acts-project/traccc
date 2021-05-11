enable_language(CUDA)

add_library(
    traccc_cuda
    SHARED
    src/component_connection.cpp
)

set_property(
    TARGET traccc_cuda
    PROPERTY
    CUDA_ARCHITECTURES 60 72 75
)

target_include_directories(
    traccc_cuda
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(
    traccc_cuda
    traccc::core
    vecmem::core
    vecmem::cuda
)

add_library(
    traccc::cuda
    ALIAS traccc_cuda
)
