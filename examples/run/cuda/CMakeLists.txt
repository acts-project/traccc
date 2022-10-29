# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cuda )

traccc_add_executable( seq_example_cuda "seq_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance 
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options)
traccc_add_executable( seeding_example_cuda "seeding_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::cuda traccc::options)
