# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  set_source_files_properties(seq_example_alpaka.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(seeding_example_alpaka.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(alpaka_ACC_GPU_CUDA_ENABLE)
    traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
       LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
       traccc::core traccc::device_common traccc::cuda traccc::alpaka alpaka::alpaka
       traccc::options )
    traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
        LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
        traccc::core traccc::device_common traccc::cuda traccc::alpaka alpaka::alpaka
        traccc::options )
else()
    traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
       LINK_LIBRARIES vecmem::core traccc::io traccc::performance
       traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
       traccc::options )
    traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
        LINK_LIBRARIES vecmem::core traccc::io traccc::performance
        traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
        traccc::options )
endif()

