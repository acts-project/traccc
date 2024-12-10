# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Find Smatrix.
find_package(ROOT COMPONENTS Core MathCore Smatrix REQUIRED)

# Set up the library.
algebra_add_library(algebra_smatrix_math smatrix_math
   "include/algebra/math/smatrix.hpp"
   "include/algebra/math/impl/smatrix_errorcheck.hpp"
   "include/algebra/math/impl/smatrix_getter.hpp"
   "include/algebra/math/impl/smatrix_matrix.hpp"
   "include/algebra/math/impl/smatrix_transform3.hpp"
   "include/algebra/math/impl/smatrix_vector.hpp")
target_link_libraries(algebra_smatrix_math
   INTERFACE algebra::common ROOT::Core ROOT::MathCore ROOT::Smatrix)
algebra_test_public_headers( algebra_smatrix_math
   "algebra/math/smatrix.hpp" )
