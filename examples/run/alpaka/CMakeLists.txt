# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# To reduce duplication of library calls, just have an empty list
# with the CUDA libs optionally added to, when needed
set(CUDA_LIBS "")

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  set_source_files_properties(seq_example_alpaka.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(seeding_example_alpaka.cpp PROPERTIES LANGUAGE CUDA)

  set_source_files_properties(full_chain_algorithm.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(throughput_st.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(throughput_mt.cpp PROPERTIES LANGUAGE CUDA)

  list(APPEND CUDA_LIBS vecmem::cuda traccc::cuda)
endif()

traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
   LINK_LIBRARIES vecmem::core traccc::io traccc::performance
   traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
   traccc::options ${CUDA_LIBS})
traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
    traccc::options ${CUDA_LIBS})

#
# Set up the "throughput applications".
#
add_library( traccc_examples_alpaka STATIC
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_alpaka
   PUBLIC vecmem::core traccc::core traccc::device_common
          traccc::alpaka alpaka::alpaka ${CUDA_LIBS})

traccc_add_executable( throughput_st_alpaka "throughput_st.cpp"
   LINK_LIBRARIES vecmem::core traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::alpaka
                  alpaka::alpaka traccc::options ${CUDA_LIBS}
                  traccc_examples_alpaka)

traccc_add_executable( throughput_mt_alpaka "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb vecmem::core traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::alpaka
                  alpaka::alpaka traccc::options ${CUDA_LIBS}
                  traccc_examples_alpaka )
