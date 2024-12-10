cmake_minimum_required( VERSION 3.28.3 )
project( CheckHIP LANGUAGES CXX )
list( INSERT CMAKE_MODULE_PATH 0       "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src/cmake"       "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src/cmake/hip"       "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src/cmake/sycl" )
enable_language( HIP )
file( WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
   "set( CMAKE_HIP_COMPILER \"${CMAKE_HIP_COMPILER}\" )" )