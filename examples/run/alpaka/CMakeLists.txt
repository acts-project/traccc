# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

set(EXTRA_LIBS)

set(TRACCC_ALPAKA_EXAMPLE_SOURCES
    seq_example_alpaka.cpp
    seeding_example_alpaka.cpp
)

include(traccc-alpaka-functions)
traccc_enable_language_alpaka()

if(alpaka_ACC_GPU_CUDA_ENABLE)
  list (APPEND EXTRA_LIBS vecmem::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  find_package( HIPToolkit REQUIRED )
  list(APPEND EXTRA_LIBS HIP::hiprt vecmem::hip)
elseif(alpaka_ACC_SYCL_ENABLE)
  list(APPEND EXTRA_LIBS vecmem::sycl)
endif()

set(LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
    traccc::options ${EXTRA_LIBS})

traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )
traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )
