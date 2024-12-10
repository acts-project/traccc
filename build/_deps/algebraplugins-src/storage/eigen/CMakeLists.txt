# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_eigen_storage eigen_storage
   "include/algebra/storage/eigen.hpp"
   "include/algebra/storage/impl/eigen_array.hpp" )
target_link_libraries( algebra_eigen_storage
   INTERFACE algebra::common Eigen3::Eigen )
algebra_test_public_headers( algebra_eigen_storage
   "algebra/storage/eigen.hpp" )
