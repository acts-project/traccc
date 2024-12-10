# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vc_math vc_math
   "include/algebra/math/vc.hpp"
   "include/algebra/math/impl/vc_getter.hpp"
   "include/algebra/math/impl/vc_transform3.hpp"
   "include/algebra/math/impl/vc_vector.hpp" )
target_link_libraries( algebra_vc_math
   INTERFACE Vc::Vc algebra::common algebra::common_math
             algebra::cmath_math )
algebra_test_public_headers( algebra_vc_math
   "algebra/math/vc.hpp" )
