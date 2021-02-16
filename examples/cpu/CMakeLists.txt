add_executable (tf_single_module tf_single_module.cpp)
target_link_libraries (tf_single_module LINK_PUBLIC traccc::core Taskflow)

add_executable (seq_tml_barrel seq_tml_barrel.cpp)
target_link_libraries (seq_tml_barrel LINK_PUBLIC traccc::core traccc::io)

add_executable (seq_tml_barrel_truth_clusters seq_tml_barrel_truth_clusters.cpp)
target_link_libraries (seq_tml_barrel_truth_clusters LINK_PUBLIC traccc::core traccc::io)