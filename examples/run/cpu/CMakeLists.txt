# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable( seeding_example "seeding_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io 
   traccc::performance traccc::options)

traccc_add_executable( seq_example "seq_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io 
   traccc::performance traccc::options)

traccc_add_executable( throughput_seq_example "throughput_seq_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io 
   traccc::performance traccc::options)

traccc_add_executable( ccl_example "ccl_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io)

traccc_add_executable( tbb_task_example "tbb_task_example.cpp"
   LINK_LIBRARIES TBB::tbb )
