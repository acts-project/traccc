cmake_minimum_required( VERSION 3.28.3 )
project( CheckSYCL LANGUAGES CXX )
list( INSERT CMAKE_MODULE_PATH 0       "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src/cmake"       "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src/cmake/hip"       "/home/slobod/Documents/Work/traccc/build/_deps/vecmem-src/cmake/sycl" )
enable_language( SYCL )
file( WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
   "set( CMAKE_SYCL_COMPILER \"${CMAKE_SYCL_COMPILER}\" )" )