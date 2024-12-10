# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include whatever CMake can give us to configure the LLVM based intel compiler,
# and use it.
include( Platform/Linux-IntelLLVM OPTIONAL
   RESULT_VARIABLE LinuxIntelLLVM_AVAILABLE )
include( Compiler/IntelLLVM OPTIONAL
   RESULT_VARIABLE IntelLLVM_AVAILABLE )
if( LinuxIntelLLVM_AVAILABLE AND IntelLLVM_AVAILABLE )
   # We have a "new enough" version of CMake to use the most appropriate
   # configuration.
   __linux_compiler_intel_llvm( SYCL )
   __compiler_intel_llvm( SYCL )
else()
   # We have a somewhat older version of CMake. Use the configuration for the
   # older Intel compilers.
   include( Platform/Linux-Intel OPTIONAL
      RESULT_VARIABLE LinuxIntel_AVAILABLE )
   include( Compiler/Intel OPTIONAL
      RESULT_VARIABLE LinuxIntel_AVAILABLE )
   if( LinuxIntel_AVAILABLE AND Intel_AVAILABLE )
      __linux_compiler_intel( SYCL )
      __compiler_intel( SYCL )
   else()
      message( WARNING "No Intel compiler configuration found in CMake! "
         "Setting fragile defaults by hand..." )
      set( CMAKE_SYCL_FLAGS_DEBUG_INIT "${CMAKE_CXX_FLAGS_DEBUG_INIT}" )
      set( CMAKE_SYCL_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT}" )
      set( CMAKE_SYCL_FLAGS_RELWITHDEBINFO_INIT
         "${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT}" )
   endif()
   # Set some options by hand, which new CMake versions do automatically.
   set( CMAKE_SYCL_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
   set( CMAKE_SYCL_COMPILE_OPTIONS_PIE "${CMAKE_CXX_COMPILE_OPTIONS_PIE}" )
endif()

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

# Set the flags controlling the C++ standard used by the SYCL compiler.
set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-std=c++17" )

set( CMAKE_SYCL20_STANDARD_COMPILE_OPTION "-std=c++20" )
set( CMAKE_SYCL20_EXTENSION_COMPILE_OPTION "-std=c++20" )

set( CMAKE_SYCL23_STANDARD_COMPILE_OPTION "-std=c++23" )
set( CMAKE_SYCL23_EXTENSION_COMPILE_OPTION "-std=c++23" )
