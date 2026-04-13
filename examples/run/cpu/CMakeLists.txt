# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

#
# Set up the "throughput applications".
#
traccc_add_library( traccc_examples_cpu examples_cpu SHARED
   "include/traccc/examples/full_chain_algorithm.hpp"
   "src/full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_cpu
   PUBLIC vecmem::core detray::core detray::detectors traccc::core
   traccc::examples_common )

traccc_add_executable( throughput_st "src/throughput_st.cpp"
   LINK_LIBRARIES indicators::indicators vecmem::core detray::detectors
                  detray::io traccc::core traccc::io traccc::performance
                  traccc::options traccc::examples_cpu )

traccc_add_executable( throughput_mt "src/throughput_mt.cpp"
   LINK_LIBRARIES indicators::indicators TBB::tbb vecmem::core detray::detectors
                  detray::io traccc::core traccc::io traccc::performance
                  traccc::options traccc::examples_cpu )

traccc_add_executable( seeding_example "src/seeding_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io
   traccc::performance traccc::options detray::detectors detray::io
   traccc::examples_common )

traccc_add_executable( seq_example "src/seq_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io
   traccc::performance traccc::options detray::detectors detray::io
   traccc::examples_common )

traccc_add_executable( truth_finding_example "src/truth_finding_example.cpp"
   LINK_LIBRARIES vecmem::core detray::detectors traccc::core traccc::io
   traccc::performance traccc::options traccc::examples_common )

traccc_add_executable( truth_fitting_example "src/truth_fitting_example.cpp"
   LINK_LIBRARIES vecmem::core detray::io detray::detectors traccc::core
   traccc::io traccc::performance traccc::options traccc::examples_common )

traccc_add_executable( misaligned_truth_fitting_example "src/misaligned_truth_fitting_example.cpp"
   LINK_LIBRARIES vecmem::core detray::io detray::detectors traccc::core
   traccc::io traccc::performance traccc::options traccc::examples_common )
