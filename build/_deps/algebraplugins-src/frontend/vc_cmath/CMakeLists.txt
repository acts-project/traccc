# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vc_cmath vc_cmath
   "include/algebra/vc_cmath.hpp" )
target_link_libraries( algebra_vc_cmath
   INTERFACE algebra::common algebra::vc_storage algebra::cmath_math
             algebra::vc_math )
algebra_test_public_headers( algebra_vc_cmath
   "algebra/vc_cmath.hpp" )
