# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library(algebra_eigen_math eigen_math
   "include/algebra/math/eigen.hpp"
   "include/algebra/math/impl/eigen_getter.hpp"
   "include/algebra/math/impl/eigen_matrix.hpp"
   "include/algebra/math/impl/eigen_transform3.hpp"
   "include/algebra/math/impl/eigen_vector.hpp")
target_link_libraries(algebra_eigen_math
   INTERFACE Eigen3::Eigen algebra::common algebra::common_math
             algebra::eigen_storage)
algebra_test_public_headers( algebra_eigen_math
   "algebra/math/eigen.hpp" )
