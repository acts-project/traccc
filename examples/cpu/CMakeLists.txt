add_executable (seq_single_module seq_single_module.cpp)
target_link_libraries (seq_single_module LINK_PUBLIC traccc::core vecmem::core)

add_executable (seq_example seq_example.cpp)
target_link_libraries (seq_example LINK_PUBLIC traccc::core traccc::io vecmem::core)

add_executable (seeding_example seeding_example.cpp)
target_link_libraries (seeding_example LINK_PUBLIC traccc::core traccc::io vecmem::core)

add_executable (ccl_example ccl_example.cpp)
target_link_libraries (ccl_example LINK_PUBLIC traccc::core traccc::io)
