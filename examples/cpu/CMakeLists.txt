add_executable (tf_single_module tf_single_module.cpp)

target_link_libraries (tf_single_module LINK_PUBLIC traccc::core Taskflow)