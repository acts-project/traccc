# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# VecMem include(s).
include( vecmem-check-language )
include( vecmem-check-sycl-source-compiles )

# CMake include(s).
include( CheckCXXSourceCompiles )

# Helper function setting up the public definitions for vecmem::sycl
#
# Usage: vecmem_setup_sycl( vecmem::sycl )
#
function( vecmem_setup_sycl libName )

   # Check if a SYCL compiler is available.
   vecmem_check_language( SYCL )

   # Set the SYCL definitions only if the build system knows about SYCL.
   if( CMAKE_SYCL_COMPILER )

      # Check if sycl::local_accessor is available.
      vecmem_check_sycl_source_compiles( "
         #include <sycl/sycl.hpp>
         int main() {
             ::sycl::queue queue;
             queue.submit([](::sycl::handler& h) {
                 ::sycl::local_accessor<int> dummy(10, h);
                 (void)dummy;
             }).wait_and_throw();
             return 0;
         }
         " VECMEM_HAVE_SYCL_LOCAL_ACCESSOR )
      if( VECMEM_HAVE_SYCL_LOCAL_ACCESSOR )
         target_compile_definitions( ${libName} INTERFACE
            $<BUILD_INTERFACE:VECMEM_HAVE_SYCL_LOCAL_ACCESSOR> )
      endif()

   endif()

endfunction( vecmem_setup_sycl )
