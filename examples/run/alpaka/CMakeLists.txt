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

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  include( traccc-compiler-options-cuda )
  set_source_files_properties(${TRACCC_ALPAKA_EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)

  list (APPEND EXTRA_LIBS vecmem::cuda traccc::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  enable_language(HIP)
  find_package( HIPToolkit REQUIRED )
  set_source_files_properties(${TRACCC_ALPAKA_EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)
  list(APPEND EXTRA_LIBS HIP::hiprt vecmem::hip)
endif()

set(LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
    traccc::options ${EXTRA_LIBS})

traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )
traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )

#Can only do this once target is defined, so need another if here
if(alpaka_ACC_GPU_HIP_ENABLE)
  set_target_properties( traccc_seq_example_alpaka PROPERTIES
     POSITION_INDEPENDENT_CODE TRUE )
  set_target_properties( traccc_seeding_example_alpaka PROPERTIES
     POSITION_INDEPENDENT_CODE TRUE )
endif()
