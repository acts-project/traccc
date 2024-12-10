# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_array_cmath array_cmath
   "include/algebra/array_cmath.hpp" )
target_link_libraries( algebra_array_cmath
   INTERFACE algebra::common algebra::array_storage algebra::cmath_math )
algebra_test_public_headers( algebra_array_cmath
   "algebra/array_cmath.hpp" )
