# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_fastor_fastor fastor_fastor
   "include/algebra/fastor_fastor.hpp" )
target_link_libraries( algebra_fastor_fastor
   INTERFACE algebra::common algebra::fastor_storage algebra::cmath_math
   			 algebra::fastor_math )
algebra_test_public_headers( algebra_fastor_fastor
   "algebra/fastor_fastor.hpp" )
