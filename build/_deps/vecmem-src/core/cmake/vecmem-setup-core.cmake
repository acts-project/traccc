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

# Helper function setting up the public definitions for vecmem::core
#
# Usage: vecmem_setup_core( vecmem::core )
#
function( vecmem_setup_core libName )

   # Check if vecmem::posix_device_atomic_ref is usable.
   get_target_property( _coreIncludes "${libName}"
      INTERFACE_INCLUDE_DIRECTORIES )
   set( CMAKE_REQUIRED_INCLUDES ${_coreIncludes} )
   unset( _coreIncludes )
   check_cxx_source_compiles( "
      #include <vecmem/memory/details/posix_device_atomic_ref.hpp>
      int main() {
         int foo = 0;
         vecmem::posix_device_atomic_ref<int> ref{foo};
         return 0;
      }
      " VECMEM_SUPPORT_POSIX_ATOMIC_REF )
   if( VECMEM_SUPPORT_POSIX_ATOMIC_REF )
      target_compile_definitions( ${libName} INTERFACE
         $<BUILD_INTERFACE:VECMEM_SUPPORT_POSIX_ATOMIC_REF> )
   endif()
   unset( CMAKE_REQUIRED_INCLUDES )

   # Check if a SYCL compiler is available.
   vecmem_check_language( SYCL )

   # Set the SYCL definitions only if the build system knows about SYCL.
   if( CMAKE_SYCL_COMPILER )

      # Test which SYCL printf function(s) is/are available.
      vecmem_check_sycl_source_compiles( "
         #include <sycl/sycl.hpp>
         #ifdef __SYCL_DEVICE_ONLY__
         #  define VECMEM_MSG_ATTRIBUTES __attribute__((opencl_constant))
         #else
         #  define VECMEM_MSG_ATTRIBUTES
         #endif
         int main() {
             const VECMEM_MSG_ATTRIBUTES char __msg[] = \"Test message %i\";
             ::sycl::ext::oneapi::experimental::printf(__msg, 20);
             return 0;
         }
         " VECMEM_HAVE_SYCL_EXT_ONEAPI_PRINTF )
      vecmem_check_sycl_source_compiles( "
         #include <sycl/sycl.hpp>
         #ifdef __SYCL_DEVICE_ONLY__
         #  define VECMEM_MSG_ATTRIBUTES __attribute__((opencl_constant))
         #else
         #  define VECMEM_MSG_ATTRIBUTES
         #endif
         int main() {
             const VECMEM_MSG_ATTRIBUTES char __msg[] = \"Test message %i\";
             ::sycl::ONEAPI::experimental::printf(__msg, 20);
             return 0;
         }
         " VECMEM_HAVE_SYCL_ONEAPI_PRINTF )

      # Set up the appropriate flag based on these checks.
      if( VECMEM_HAVE_SYCL_EXT_ONEAPI_PRINTF )
         target_compile_definitions( ${libName} INTERFACE
            $<BUILD_INTERFACE:VECMEM_SYCL_PRINTF_FUNCTION=::sycl::ext::oneapi::experimental::printf> )
      elseif( VECMEM_HAVE_SYCL_ONEAPI_PRINTF )
         target_compile_definitions( ${libName} INTERFACE
            $<BUILD_INTERFACE:VECMEM_SYCL_PRINTF_FUNCTION=::sycl::ONEAPI::experimental::printf> )
      else()
         message( WARNING "No valid printf function found for SYCL."
            " Enabling debug messages will likely not work in device code." )
         target_compile_definitions( ${libName} INTERFACE
            $<BUILD_INTERFACE:VECMEM_SYCL_PRINTF_FUNCTION=printf>
            $<BUILD_INTERFACE:VECMEM_MSG_ATTRIBUTES=> )
      endif()

      # Test whether sycl::atomic_ref is available.
      vecmem_check_sycl_source_compiles( "
         #include <sycl/sycl.hpp>
         int main() {
             int dummy = 0;
             ::sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               ::sycl::memory_scope::device,
                               ::sycl::access::address_space::global_space>
                 atomic_dummy(dummy);
             atomic_dummy.store(3);
             atomic_dummy.fetch_add(1);
             return 0;
         }
         " VECMEM_HAVE_SYCL_ATOMIC_REF )
      if( VECMEM_HAVE_SYCL_ATOMIC_REF )
         target_compile_definitions( ${libName} INTERFACE
            $<BUILD_INTERFACE:VECMEM_HAVE_SYCL_ATOMIC_REF> )
      endif()

   endif()

endfunction( vecmem_setup_core )
