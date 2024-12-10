# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Use the standard GNU compiler options for ComputeCpp.
include( Platform/Linux-GNU )
__linux_compiler_gnu( SYCL )
include( Compiler/GNU )
__compiler_gnu( SYCL )

# Set up the dependency file generation for this platform. Note that SYCL
# compilation only works with Makefile and Ninja generators, so no check is made
# here for the current generator.
set( CMAKE_SYCL_DEPENDS_USE_COMPILER TRUE )
set( CMAKE_SYCL_DEPFILE_FORMAT gcc )

# Set a compiler command from scratch for this platform.
set( CMAKE_SYCL_COMPILE_OBJECT
   "<CMAKE_SYCL_COMPILER> -x c++ <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>" )

# Set an archive (static library) creation command explicitly for this platform.
set( CMAKE_SYCL_CREATE_STATIC_LIBRARY
   "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>" )

# Find the ComputeCpp library, and set it up to be linked to anything that is
# built by this compiler.
find_library( CMAKE_SYCL_STANDARD_LIBRARIES
   NAMES ComputeCpp
   DOC "ComputeCpp runtime library" )
mark_as_advanced( CMAKE_SYCL_STANDARD_LIBRARIES )

# Set the flags controlling the C++ standard used by the SYCL compiler.
set( CMAKE_SYCL11_STANDARD_COMPILE_OPTION "-std=c++11" )
set( CMAKE_SYCL11_EXTENSION_COMPILE_OPTION "-std=c++11" )

set( CMAKE_SYCL14_STANDARD_COMPILE_OPTION "-std=c++14" )
set( CMAKE_SYCL14_EXTENSION_COMPILE_OPTION "-std=c++14" )

set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-std=c++17" )

set( CMAKE_SYCL20_STANDARD_COMPILE_OPTION "-std=c++20" )
set( CMAKE_SYCL20_EXTENSION_COMPILE_OPTION "-std=c++20" )

set( CMAKE_SYCL23_STANDARD_COMPILE_OPTION "-std=c++23" )
set( CMAKE_SYCL23_EXTENSION_COMPILE_OPTION "-std=c++23" )
