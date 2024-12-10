# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library(algebra_common_math common_math
   # Math
   "include/algebra/math/common.hpp")
target_link_libraries(algebra_common_math
   INTERFACE algebra::common)
algebra_test_public_headers( algebra_common_math
   "algebra/math/common.hpp" )
