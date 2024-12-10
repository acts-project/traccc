# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_fastor_storage fastor_storage
   "include/algebra/storage/fastor.hpp"
   "include/algebra/storage/impl/fastor_matrix.hpp" )
target_link_libraries( algebra_fastor_storage
	INTERFACE Fastor::Fastor algebra::common )
algebra_test_public_headers( algebra_fastor_storage
   "algebra/storage/fastor.hpp" )
