# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  set_source_files_properties(seeding_example_alpaka.cpp clustering_example_alpaka.cpp alpakaHWTest.cpp alpakaVectorAdd.cpp alpakaBufferCopy.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(alpaka_ACC_GPU_CUDA_ENABLE)
    traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
        LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
        traccc::core traccc::device_common traccc::cuda traccc::alpaka alpaka::alpaka
        traccc::options )
      
    traccc_add_executable( clustering_example_alpaka "clustering_example_alpaka.cpp"
        LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
        traccc::core traccc::device_common traccc::cuda traccc::alpaka alpaka::alpaka 
        traccc::options ) 
    
    traccc_add_executable( alpakaHWTest "alpakaHWTest.cpp"
        LINK_LIBRARIES alpaka::alpaka traccc::cuda)

    traccc_add_executable( alpakaVectorAdd "alpakaVectorAdd.cpp"
        LINK_LIBRARIES alpaka::alpaka traccc::cuda )

    traccc_add_executable( alpakaBufferCopy "alpakaBufferCopy.cpp"
        LINK_LIBRARIES alpaka::alpaka traccc::cuda )
    
else()
    traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
        LINK_LIBRARIES vecmem::core traccc::io traccc::performance
        traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
        traccc::options )

    traccc_add_executable( clustering_example_alpaka "clustering_example_alpaka.cpp"
        LINK_LIBRARIES vecmem::core traccc::io traccc::performance
        traccc::core traccc::device_common traccc::alpaka alpaka::alpaka 
        traccc::options )
endif()

