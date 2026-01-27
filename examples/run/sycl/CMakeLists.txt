# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2026 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# SYCL seeding executable(s).
traccc_add_executable( seeding_example_sycl "seeding_example_sycl.cpp"
   LINK_LIBRARIES traccc::options vecmem::core vecmem::sycl traccc::io
                  traccc::core traccc::device_common traccc::sycl
                  traccc::performance traccc_examples_common )

traccc_add_executable( seq_example_sycl "seq_example_sycl.cpp"
   LINK_LIBRARIES traccc::options vecmem::core vecmem::sycl traccc::io
                  traccc::core traccc::device_common traccc::sycl
                  traccc::performance traccc_examples_common )

#
# Set up the "throughput applications".
#
add_library( traccc_examples_sycl STATIC
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_sycl
   PUBLIC vecmem::core vecmem::sycl detray::core detray::detectors
          traccc::core traccc::device_common traccc::sycl
          traccc_examples_common )

traccc_add_executable( throughput_st_sycl "throughput_st.cpp"
   LINK_LIBRARIES indicators::indicators vecmem::core vecmem::sycl
                  detray::detectors detray::io traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::sycl
                  traccc::options traccc_examples_sycl )

traccc_add_executable( throughput_mt_sycl "throughput_mt.cpp"
   LINK_LIBRARIES indicators::indicators TBB::tbb vecmem::core vecmem::sycl
                  detray::detectors detray::io traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::sycl
                  traccc::options traccc_examples_sycl )
