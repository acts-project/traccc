# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_common common
   "include/algebra/qualifiers.hpp" )
algebra_test_public_headers( algebra_common
   "algebra/qualifiers.hpp" )
