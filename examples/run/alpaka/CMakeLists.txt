# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-alpaka-functions )

set(EXTRA_LIBS)

if(alpaka_ACC_GPU_CUDA_ENABLE)
  list (APPEND EXTRA_LIBS vecmem::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  find_package( HIPToolkit REQUIRED )
  list(APPEND EXTRA_LIBS HIP::hiprt vecmem::hip)
elseif(alpaka_ACC_SYCL_ENABLE)
  list(APPEND EXTRA_LIBS vecmem::sycl)
endif()

set(LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka
    traccc::options traccc_examples_common ${EXTRA_LIBS})
set(DETRAY detray::io detray::detectors)

traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} ${DETRAY} )
traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )

#
# Set up the "throughput applications".
#
add_library( traccc_examples_alpaka STATIC
   "device_backend.hpp"
   "device_backend.cpp"
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_alpaka
   PUBLIC vecmem::core detray::core detray::detectors
   traccc::core traccc::device_common traccc::alpaka traccc_examples_common ${EXTRA_LIBS})

traccc_add_executable( throughput_st_alpaka "throughput_st.cpp"
   LINK_LIBRARIES indicators::indicators ${LIBRARIES} ${DETRAY} traccc_examples_alpaka )

traccc_add_executable( throughput_mt_alpaka "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb indicators::indicators ${LIBRARIES} ${DETRAY} traccc_examples_alpaka )

traccc_add_executable( track_finding_validation_alpaka
   "track_finding_validation.cpp"
   LINK_LIBRARIES traccc_examples_common traccc_examples_alpaka )
