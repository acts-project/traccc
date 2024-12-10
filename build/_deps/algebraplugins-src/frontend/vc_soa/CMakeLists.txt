# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vc_soa vc_soa
   "include/algebra/vc_soa.hpp" )
target_link_libraries( algebra_vc_soa
   INTERFACE algebra::common algebra::vc_soa_storage algebra::cmath_math
             algebra::vc_soa_math )
algebra_test_public_headers( algebra_vc_soa
   "algebra/vc_soa.hpp" )
