# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vc_vc vc_vc
   "include/algebra/vc_vc.hpp" )
target_link_libraries( algebra_vc_vc
   INTERFACE algebra::common algebra::vc_storage algebra::cmath_math
             algebra::vc_math )
algebra_test_public_headers( algebra_vc_vc
   "algebra/vc_vc.hpp" )
