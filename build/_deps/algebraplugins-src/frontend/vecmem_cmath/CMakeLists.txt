# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vecmem_cmath vecmem_cmath
   "include/algebra/vecmem_cmath.hpp" )
target_link_libraries( algebra_vecmem_cmath
   INTERFACE algebra::common algebra::vecmem_storage algebra::cmath_math )
algebra_test_public_headers( algebra_vecmem_cmath
   "algebra/vecmem_cmath.hpp" )
