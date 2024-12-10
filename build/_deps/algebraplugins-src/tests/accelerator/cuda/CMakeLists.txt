# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake version requirement.
cmake_minimum_required( VERSION 3.17 )

# Enable CUDA as a language.
enable_language( CUDA )
find_package( CUDAToolkit REQUIRED )

# Set the default CUDA compiler flags.
include( algebra-plugins-compiler-options-cuda )

# Helper library for the CUDA tests.
add_library( algebra_tests_cuda_common STATIC
   "common/cuda_error_check.hpp" "common/cuda_error_check.cpp"
   "common/execute_cuda_test.cuh" "common/test_cuda_basics.cuh" )
target_include_directories( algebra_tests_cuda_common PUBLIC
   "${CMAKE_CURRENT_SOURCE_DIR}/common" )
target_link_libraries( algebra_tests_cuda_common
   PUBLIC    CUDA::cudart
   INTERFACE GTest::gtest algebra::common algebra::tests_common
             algebra::tests_accel_common vecmem::core vecmem::cuda )
target_compile_options( algebra_tests_cuda_common INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=177>
   $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false --prec-div=true>
   $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true --fmad=false> )
if( "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "11.5" )
   # Replace all "#pragma diag_suppress" calls in the Eigen code with
   # "#pragma nv_diag_suppress" calls. To be removed once the Eigen
   # code starts behaving better. Note that this is only needed for
   # Windows, as these warnings are suppressed on Linux by declaring
   # Eigen as a "system include".
   target_compile_definitions( algebra_tests_cuda_common
      INTERFACE diag_suppress=nv_diag_suppress )
endif()
add_library( algebra::tests_cuda_common ALIAS algebra_tests_cuda_common )

# Set up all of the (available) CUDA tests.
algebra_add_test( array_cuda "array_cmath.cu"
   LINK_LIBRARIES algebra::array_cmath
                  algebra::tests_cuda_common GTest::gtest_main )
target_compile_options( algebra_test_array_cuda
   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> )

if( ALGEBRA_PLUGINS_INCLUDE_EIGEN )
   algebra_add_test( eigen_cuda "eigen_cmath.cu" "eigen_eigen.cu"
      LINK_LIBRARIES algebra::eigen_cmath algebra::eigen_eigen
                     algebra::tests_cuda_common GTest::gtest_main )
   target_compile_options( algebra_test_eigen_cuda
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_VECMEM )
   algebra_add_test( vecmem_cuda "vecmem_cmath.cu"
      LINK_LIBRARIES algebra::vecmem_cmath
                     algebra::tests_cuda_common GTest::gtest_main )
endif()
