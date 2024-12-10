# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_smatrix_smatrix smatrix_smatrix
   "include/algebra/smatrix_smatrix.hpp" )
target_link_libraries( algebra_smatrix_smatrix
   INTERFACE algebra::common algebra::smatrix_storage algebra::smatrix_math )
algebra_test_public_headers( algebra_smatrix_smatrix
   "algebra/smatrix_smatrix.hpp" )
