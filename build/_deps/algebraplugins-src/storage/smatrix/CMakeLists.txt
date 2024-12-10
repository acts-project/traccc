# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Find Smatrix.
find_package( ROOT COMPONENTS Smatrix REQUIRED )

# Set up the library.
algebra_add_library( algebra_smatrix_storage smatrix_storage
   "include/algebra/storage/smatrix.hpp" )
target_link_libraries( algebra_smatrix_storage
   INTERFACE algebra::common ROOT::Smatrix )
algebra_test_public_headers( algebra_smatrix_storage
   "algebra/storage/smatrix.hpp" )
