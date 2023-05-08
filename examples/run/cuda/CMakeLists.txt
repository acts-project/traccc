# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cuda )

# External(s).
find_package( CUDAToolkit REQUIRED )

traccc_add_executable( seq_example_cuda "seq_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options )
traccc_add_executable( seeding_example_cuda "seeding_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options )

#
# Set up the "throughput applications".
#


#set (CMAKE_REQUIRED_INCLUDES "/usr/local/include")
#CHECK_INCLUDE_FILE_CXX("cxxabi.h" GLOG_INCLUDE)
#IF(GLOG_INCLUDE)
#    MESSAGE("izinto")
#ENDIF(GLOG_INCLUDE)

add_library( traccc_examples_cuda STATIC
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
traccc_add_library(nvtx_instrumentation TYPE OBJECT
        "inst_nvtx.cpp" )

include( CheckIncludeFileCXX )
check_include_file_cxx( cxxabi.h TRACCC_HAVE_CXXABI_H )
if( TRACCC_HAVE_CXXABI_H )
    target_compile_definitions( nvtx_instrumentation PRIVATE TRACCC_HAVE_CXXABI_H )
endif()

set_target_properties( nvtx_instrumentation PROPERTIES
        POSITION_INDEPENDENT_CODE ON )

target_link_libraries( traccc_examples_cuda
   PUBLIC CUDA::cudart vecmem::core vecmem::cuda traccc::core
          traccc::device_common traccc::cuda )

target_link_libraries( nvtx_instrumentation
        PUBLIC CUDA::cudart traccc::core
        traccc::device_common traccc::cuda )

traccc_add_executable( throughput_st_cuda "throughput_st.cpp"
        LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_cuda
                  nvtx_instrumentation ${CMAKE_DL_LIBS})

traccc_add_executable( throughput_mt_cuda "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_cuda )
