# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( Platform/Windows-IntelLLVM )
include( Compiler/IntelLLVM )

# Set up the variables specifying the command line arguments of the compiler.
# Unfortunately these don't really do anything at the moment, as the CMake code
# has pretty specific paths for the different languages with MSVC. :-(
__windows_compiler_intel( SYCL )
__compiler_intel_llvm( SYCL )

# Because the above doesn't do much, take the C++ flags set up on Windows, and
# use them for SYCL compilation as well.
string( APPEND CMAKE_SYCL_FLAGS_INIT ${CMAKE_CXX_FLAGS_INIT} )
string( APPEND CMAKE_SYCL_FLAGS_DEBUG_INIT ${CMAKE_CXX_FLAGS_DEBUG_INIT} )
string( APPEND CMAKE_SYCL_FLAGS_RELEASE_INIT ${CMAKE_CXX_FLAGS_RELEASE_INIT} )
string( APPEND CMAKE_SYCL_FLAGS_RELWITHDEBINFO_INIT
   ${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT} )

# Set up the dependency file generation for this platform. Note that SYCL
# compilation only works with Makefile and Ninja generators, so no check is made
# here for the current generator.
set( CMAKE_SYCL_DEPENDS_USE_COMPILER TRUE )
set( CMAKE_SYCL_DEPFILE_FORMAT gcc )

# With CMake 3.27 by default CMake would try to add version information into
# the DLL file names. Confusing the linking process.
#
# I tried to disable this using the DLL_NAME_WITH_SOVERSION property, the thing
# that was introduced in CMake 3.27, but for some reason that did not work. :-/
# So when SYCL code is being used on Windows, I disable "versioned SONAMEs"
# completely. For all languages and targets.
set( CMAKE_PLATFORM_NO_VERSIONED_SONAME TRUE )

# Tweak the compiler command, to let the compiler explicitly know that it is
# receiving C++ source code with the provided .sycl file(s).
string( REPLACE "<SOURCE>" "/Tp <SOURCE>" CMAKE_SYCL_COMPILE_OBJECT
   "${CMAKE_SYCL_COMPILE_OBJECT}" )

# Use the (basic) flags set up for compilation, during linking as well.
set( CMAKE_SYCL_LINK_FLAGS "${CMAKE_SYCL_FLAGS}" )

# Tweak the linker commands to use the DPC++ executable for linking, and to
# pass the arguments to the linker correctly.
foreach( linker_command "CMAKE_SYCL_CREATE_SHARED_LIBRARY"
   "CMAKE_SYCL_CREATE_SHARED_MODULE" "CMAKE_SYCL_LINK_EXECUTABLE" )

   # Replace the VS linker with DPC++. (Note that this does not do
   # anything with modern CMake versions anymore. As those use
   # <CMAKE_SYCL_COMPILER> in the link command out of the box.)
   string( REPLACE "<CMAKE_LINKER>" "\"${CMAKE_SYCL_HOST_LINKER}\""
      ${linker_command} "${${linker_command}}" )

   # Prefix the linker-specific arguments with "/link", to let DPC++ know
   # that these are to be given to the linker. "/out" just happens to be the
   # first linker argument on the command line. (With CMake 3.21.)
   #
   # Later CMake versions fixed this out of the box, so this is only needed
   # if "/link" is missing from the commands.
   if( ( NOT "${${linker_command}}" MATCHES "/link" ) AND
       ( NOT "${${linker_command}}" MATCHES "-link" ) )
      string( REPLACE "/out" "/link /out"
         ${linker_command} "${${linker_command}}" )
   endif()

endforeach()

# Set the flags controlling the C++ standard used by the SYCL compiler.
set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "/std:c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "/std:c++17" )

set( CMAKE_SYCL20_STANDARD_COMPILE_OPTION "/std:c++20" )
set( CMAKE_SYCL20_EXTENSION_COMPILE_OPTION "/std:c++20" )

set( CMAKE_SYCL23_STANDARD_COMPILE_OPTION "/std:c++23" )
set( CMAKE_SYCL23_EXTENSION_COMPILE_OPTION "/std:c++23" )
