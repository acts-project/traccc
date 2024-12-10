# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-check-sycl-source-compiles )

# Set up the build of the VecMem core library.
vecmem_add_library( vecmem_core core
   # STL mimicking containers.
   "include/vecmem/containers/array.hpp"
   "include/vecmem/containers/impl/array.ipp"
   "include/vecmem/containers/device_array.hpp"
   "include/vecmem/containers/impl/device_array.ipp"
   "include/vecmem/containers/device_vector.hpp"
   "include/vecmem/containers/impl/device_vector.ipp"
   "include/vecmem/containers/static_vector.hpp"
   "include/vecmem/containers/impl/static_vector.ipp"
   "include/vecmem/containers/jagged_device_vector.hpp"
   "include/vecmem/containers/impl/jagged_device_vector.ipp"
   "include/vecmem/containers/jagged_vector.hpp"
   "include/vecmem/containers/impl/jagged_vector.ipp"
   "include/vecmem/containers/vector.hpp"
   "include/vecmem/containers/impl/vector.ipp"
   # Data holding/transporting types.
   "include/vecmem/containers/data/jagged_vector_buffer.hpp"
   "include/vecmem/containers/impl/jagged_vector_buffer.ipp"
   "include/vecmem/containers/data/jagged_vector_data.hpp"
   "include/vecmem/containers/impl/jagged_vector_data.ipp"
   "include/vecmem/containers/data/jagged_vector_view.hpp"
   "include/vecmem/containers/impl/jagged_vector_view.ipp"
   "include/vecmem/containers/data/vector_buffer.hpp"
   "include/vecmem/containers/impl/vector_buffer.ipp"
   "include/vecmem/containers/data/vector_view.hpp"
   "include/vecmem/containers/impl/vector_view.ipp"
   # Iterator types.
   "include/vecmem/containers/details/jagged_device_vector_iterator.hpp"
   "include/vecmem/containers/impl/jagged_device_vector_iterator.ipp"
   "include/vecmem/containers/details/reverse_iterator.hpp"
   "include/vecmem/containers/impl/reverse_iterator.ipp"
   # Container helper code.
   "include/vecmem/containers/details/resize_jagged_vector.hpp"
   # Allocator
   "include/vecmem/memory/allocator.hpp"
   "include/vecmem/memory/impl/allocator.ipp"
   "src/memory/allocator.cpp"
   # Memory management.
   "include/vecmem/memory/atomic.hpp"
   "include/vecmem/memory/impl/atomic.ipp"
   "include/vecmem/memory/polymorphic_allocator.hpp"
   "include/vecmem/memory/memory_resource.hpp"
   "include/vecmem/memory/details/unique_alloc_deleter.hpp"
   "include/vecmem/memory/details/unique_obj_deleter.hpp"
   "include/vecmem/memory/unique_ptr.hpp"
   "include/vecmem/memory/details/is_aligned.hpp"
   "src/memory/details/is_aligned.cpp"
   # Atomic reference(s).
   "include/vecmem/memory/device_address_space.hpp"
   "include/vecmem/memory/device_atomic_ref.hpp"
   "include/vecmem/memory/details/dummy_device_atomic_ref.hpp"
   "include/vecmem/memory/impl/dummy_device_atomic_ref.ipp"
   "include/vecmem/memory/details/cuda_device_atomic_ref.hpp"
   "include/vecmem/memory/impl/cuda_device_atomic_ref.ipp"
   "include/vecmem/memory/details/hip_device_atomic_ref.hpp"
   "include/vecmem/memory/details/sycl_builtin_device_atomic_ref.hpp"
   "include/vecmem/memory/details/sycl_custom_device_atomic_ref.hpp"
   "include/vecmem/memory/impl/sycl_custom_device_atomic_ref.ipp"
   "include/vecmem/memory/details/posix_device_atomic_ref.hpp"
   "include/vecmem/memory/impl/posix_device_atomic_ref.ipp"
   # EDM types.
   "include/vecmem/edm/container.hpp"
   "include/vecmem/edm/buffer.hpp"
   "include/vecmem/edm/impl/buffer.ipp"
   "include/vecmem/edm/data.hpp"
   "include/vecmem/edm/impl/data.ipp"
   "include/vecmem/edm/device.hpp"
   "include/vecmem/edm/impl/device.ipp"
   "include/vecmem/edm/host.hpp"
   "include/vecmem/edm/impl/host.ipp"
   "include/vecmem/edm/view.hpp"
   "include/vecmem/edm/impl/view.ipp"
   "include/vecmem/edm/schema.hpp"
   "include/vecmem/edm/details/buffer_traits.hpp"
   "include/vecmem/edm/details/data_traits.hpp"
   "include/vecmem/edm/details/device_traits.hpp"
   "include/vecmem/edm/details/host_traits.hpp"
   "include/vecmem/edm/details/view_traits.hpp"
   "include/vecmem/edm/details/schema_traits.hpp"
   "include/vecmem/edm/details/types.hpp"
   #
   # Memory resources.
   #
   # Memory resource base types.
   "src/memory/details/memory_resource_base.cpp"
   "include/vecmem/memory/details/memory_resource_base.hpp"
   "src/memory/details/memory_resource_impl.hpp"
   # Host memory resource.
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/host_memory_resource.hpp"
   # Arena memory resource.
   "src/memory/details/arena_memory_resource_impl.cpp"
   "src/memory/details/arena_memory_resource_impl.hpp"
   "src/memory/arena_memory_resource.cpp"
   "include/vecmem/memory/arena_memory_resource.hpp"
   # Binary page memory resource.
   "src/memory/details/binary_page_memory_resource_impl.cpp"
   "src/memory/details/binary_page_memory_resource_impl.hpp"
   "src/memory/binary_page_memory_resource.cpp"
   "include/vecmem/memory/binary_page_memory_resource.hpp"
   # Pool memory resource.
   "src/memory/details/pool_memory_resource_impl.cpp"
   "src/memory/details/pool_memory_resource_impl.hpp"
   "src/memory/pool_memory_resource.cpp"
   "include/vecmem/memory/pool_memory_resource.hpp"
   # Choice memory resource.
   "src/memory/details/choice_memory_resource_impl.cpp"
   "src/memory/details/choice_memory_resource_impl.hpp"
   "src/memory/choice_memory_resource.cpp"
   "include/vecmem/memory/choice_memory_resource.hpp"
   # Coalescing memory resource.
   "src/memory/details/coalescing_memory_resource_impl.cpp"
   "src/memory/details/coalescing_memory_resource_impl.hpp"
   "src/memory/coalescing_memory_resource.cpp"
   "include/vecmem/memory/coalescing_memory_resource.hpp"
   # Conditional memory resource.
   "src/memory/details/conditional_memory_resource_impl.cpp"
   "src/memory/details/conditional_memory_resource_impl.hpp"
   "src/memory/conditional_memory_resource.cpp"
   "include/vecmem/memory/conditional_memory_resource.hpp"
   # Contiguous memory resource.
   "src/memory/details/contiguous_memory_resource_impl.cpp"
   "src/memory/details/contiguous_memory_resource_impl.hpp"
   "src/memory/contiguous_memory_resource.cpp"
   "include/vecmem/memory/contiguous_memory_resource.hpp"
   # Debug memory resource.
   "src/memory/details/debug_memory_resource_impl.cpp"
   "src/memory/details/debug_memory_resource_impl.hpp"
   "src/memory/debug_memory_resource.cpp"
   "include/vecmem/memory/debug_memory_resource.hpp"
   # Identity memory resource.
   "src/memory/identity_memory_resource.cpp"
   "include/vecmem/memory/identity_memory_resource.hpp"
   # Instrumenting memory resource.
   "src/memory/details/instrumenting_memory_resource_impl.cpp"
   "src/memory/details/instrumenting_memory_resource_impl.hpp"
   "src/memory/instrumenting_memory_resource.cpp"
   "include/vecmem/memory/instrumenting_memory_resource.hpp"
   # Terminal memory resource.
   "src/memory/terminal_memory_resource.cpp"
   "include/vecmem/memory/terminal_memory_resource.hpp"
   # Synchronized memory resource.
   "src/memory/synchronized_memory_resource.cpp"
   "include/vecmem/memory/synchronized_memory_resource.hpp"
   # Utilities.
   "include/vecmem/utils/copy.hpp"
   "include/vecmem/utils/impl/copy.ipp"
   "src/utils/copy.cpp"
   "include/vecmem/utils/debug.hpp"
   "src/utils/memory_monitor.cpp"
   "include/vecmem/utils/memory_monitor.hpp"
   "include/vecmem/utils/tuple.hpp"
   "include/vecmem/utils/impl/tuple.ipp"
   "include/vecmem/utils/type_traits.hpp"
   "include/vecmem/utils/types.hpp"
   "src/utils/integer_math.hpp" )

# Hide the library's symbols by default.
set_target_properties( vecmem_core PROPERTIES
   CXX_VISIBILITY_PRESET "hidden" )
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag(
   "-fvisibility-ms-compat" VECMEM_HAVE_VISIBILITY_MS_COMPAT )
if( VECMEM_HAVE_VISIBILITY_MS_COMPAT )
   target_compile_options( vecmem_core PRIVATE "-fvisibility-ms-compat" )
endif()

# Generate a version header for the project.
configure_file( "cmake/version.hpp.in"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem/version.hpp"
   @ONLY )
install(
   FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem/version.hpp"
   DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vecmem" )

# Add definitions necessary for the correct functioning of VECMEM_DEBUG_MSG.
string( LENGTH "${CMAKE_SOURCE_DIR}/" VECMEM_SOURCE_DIR_LENGTH )
target_compile_definitions( vecmem_core PUBLIC
   VECMEM_DEBUG_MSG_LVL=${VECMEM_DEBUG_MSG_LVL}
   $<BUILD_INTERFACE:VECMEM_SOURCE_DIR_LENGTH=${VECMEM_SOURCE_DIR_LENGTH}> )

# The library headers make checks on the value of the __cplusplus macro. So we
# need to make sure that MSVC would always set that macro up when using this
# library.
target_compile_options( vecmem_core PUBLIC
   $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/Zc:__cplusplus>
   $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler /Zc:__cplusplus> )

# Set up the public definitions for vecmem::core, which require configuration
# tests to be made with the C++/SYCL/etc. compilers.
include( "cmake/vecmem-setup-core.cmake" )
vecmem_setup_core( vecmem_core )
install( FILES "cmake/vecmem-setup-core.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )

# Figure out where to get <memory_resource> from.
include( CheckCXXSourceCompiles )
check_cxx_source_compiles( "
   #include <memory_resource>
   int main() {
       std::pmr::memory_resource* mr = nullptr;
       std::pmr::polymorphic_allocator<int> pa(mr);
       return 0;
   }
   " VECMEM_HAVE_PMR_MEMORY_RESOURCE )
   set(pmr_include "memory_resource")
   set(pmr_ns "std::pmr")
if( VECMEM_HAVE_PMR_MEMORY_RESOURCE )
   message( STATUS "Using memory resource types from the std::pmr namespace" )
   target_compile_definitions( vecmem_core PUBLIC
      VECMEM_HAVE_PMR_MEMORY_RESOURCE )
else()
   check_cxx_source_compiles( "
      #include <experimental/memory_resource>
      int main() {
          std::experimental::pmr::memory_resource* mr = nullptr;
          std::experimental::pmr::polymorphic_allocator<int> pa(mr);
          return 0;
      }
      " VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE )
   if( VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE )
      message( STATUS "Using memory resource types from the "
         "std::experimental::pmr namespace" )
      target_compile_definitions( vecmem_core PUBLIC
         VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE )

      set(pmr_include "experimental/memory_resource")
      set(pmr_ns "std::experimental::pmr")
   else()
      message( SEND_ERROR
         "C++17 LFTS V1 (P0220R1) component memory_resource not found!" )
   endif()
endif()

# Figure out whether or not to include the default resource polyfill.
check_cxx_source_compiles( "
   #include <${pmr_include}>
   int main() {
         ${pmr_ns}::set_default_resource(nullptr);
         (void)${pmr_ns}::get_default_resource();
         return 0;
   }
  " VECMEM_HAVE_DEFAULT_RESOURCE)
if(NOT VECMEM_HAVE_DEFAULT_RESOURCE)
   target_sources( vecmem_core
      PRIVATE "src/memory/default_resource_polyfill.cpp" )
   source_group( "src\\\\memory"
      FILES "src/memory/default_resource_polyfill.cpp" )
endif()

# Test whether we have CLZ builtins.
check_cxx_source_compiles( "
   #include <intrin.h>
   #include <cstddef>
   int main() {
      (void)_lzcnt_u64(static_cast<std::size_t>(1UL));
      return 0;
   }
   " VECMEM_HAVE_LZCNT_U64 )
if( VECMEM_HAVE_LZCNT_U64 )
   target_compile_definitions(
      vecmem_core
      PRIVATE VECMEM_HAVE_LZCNT_U64
   )
endif()

check_cxx_source_compiles( "
   #include <cstddef>
   int main() {
      (void)__builtin_clzl(static_cast<std::size_t>(1UL));
      return 0;
   }
   " VECMEM_HAVE_BUILTIN_CLZL )
if( VECMEM_HAVE_BUILTIN_CLZL )
   target_compile_definitions(
      vecmem_core
      PRIVATE VECMEM_HAVE_BUILTIN_CLZL
   )
endif()

# Check if std::aligned_alloc is available. With MSVC (at the time of writing)
# it is not.
include( CheckCXXSymbolExists )
check_cxx_symbol_exists( "std::aligned_alloc" "cstdlib"
   VECMEM_HAVE_STD_ALIGNED_ALLOC )
if( VECMEM_HAVE_STD_ALIGNED_ALLOC )
   target_compile_definitions( vecmem_core
      PRIVATE VECMEM_HAVE_STD_ALIGNED_ALLOC )
endif()

# Test the public headers of vecmem::core.
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
   file( GLOB vecmem_core_public_headers
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "include/vecmem/*/*.hpp" )
   list( APPEND vecmem_core_public_headers "vecmem/version.hpp" )
   vecmem_test_public_headers( vecmem_core ${vecmem_core_public_headers} )
endif()
