# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set the default C++ compiler flags.
include( algebra-plugins-compiler-options-cpp )

# Set up an INTERFACE library for the common header(s).
add_library( algebra_tests_common INTERFACE )
target_include_directories( algebra_tests_common INTERFACE
   "${CMAKE_CURRENT_SOURCE_DIR}/common" )
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   target_compile_definitions( algebra_tests_common INTERFACE
      -D_USE_MATH_DEFINES )
endif()
add_library( algebra::tests_common ALIAS algebra_tests_common )

# Set up all of the (available) "host" tests.
algebra_add_test( array
   "array/array_cmath.cpp"
   LINK_LIBRARIES GTest::gtest_main algebra::tests_common
                  algebra::array_cmath )

if( ALGEBRA_PLUGINS_INCLUDE_EIGEN )
   algebra_add_test( eigen
      "eigen/eigen_cmath.cpp" "eigen/eigen_eigen.cpp"
      LINK_LIBRARIES GTest::gtest_main algebra::tests_common
                     algebra::eigen_cmath algebra::eigen_eigen )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_SMATRIX )
   algebra_add_test( smatrix
      "smatrix/smatrix_cmath.cpp" "smatrix/smatrix_smatrix.cpp"
      LINK_LIBRARIES GTest::gtest_main algebra::tests_common
                     algebra::smatrix_cmath algebra::smatrix_smatrix )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_VC )
   algebra_add_test( vc
      "vc/vc_cmath.cpp" "vc/vc_vc.cpp"
      LINK_LIBRARIES GTest::gtest_main algebra::tests_common
                      algebra::vc_cmath algebra::vc_vc )

   if( NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang" )
      algebra_add_test( vc_soa
         "vc_soa/vc_soa.cpp"
         LINK_LIBRARIES GTest::gtest_main algebra::vc_soa )
   endif()
endif()

if( ALGEBRA_PLUGINS_INCLUDE_FASTOR )
   algebra_add_test( fastor 
      "fastor/fastor_fastor.cpp"
      LINK_LIBRARIES GTest::gtest_main algebra::tests_common
                     algebra::fastor_fastor )
endif()

if( ALGEBRA_PLUGINS_INCLUDE_VECMEM )
   algebra_add_test( vecmem
      "vecmem/vecmem_cmath.cpp"
      LINK_LIBRARIES GTest::gtest_main algebra::tests_common
                     algebra::vecmem_cmath )
endif()

# Set up all of the (available) "device" tests.
add_subdirectory( accelerator )
