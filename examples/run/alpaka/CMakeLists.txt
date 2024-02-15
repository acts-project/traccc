# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

set(EXTRA_LIBS "")

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  include( traccc-compiler-options-cuda )

  list (APPEND EXTRA_LIBS vecmem::cuda traccc::cuda)
endif()

set(LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
    traccc::options ${EXTRA_LIBS})

traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )
traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )

#
# Set up the "throughput applications".
#
alpaka_add_library( traccc_examples_alpaka SHARED
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_alpaka
   PUBLIC vecmem::core traccc::core traccc::device_common
          traccc::alpaka alpaka::alpaka ${EXTRA_LIBS})

traccc_add_executable( throughput_st_alpaka "throughput_st.cpp"
   LINK_LIBRARIES ${LIBRARIES} traccc_examples_alpaka)

traccc_add_executable( throughput_mt_alpaka "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb ${LIBRARIES} traccc_examples_alpaka )

