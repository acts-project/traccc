# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library(algebra_cmath_math cmath_math
   # impl include
   "include/algebra/math/cmath.hpp"
   "include/algebra/math/impl/cmath_getter.hpp"
   "include/algebra/math/impl/cmath_matrix.hpp"
   "include/algebra/math/impl/cmath_operators.hpp"
   "include/algebra/math/impl/cmath_transform3.hpp"
   "include/algebra/math/impl/cmath_vector.hpp"
   # algorithms include
   "include/algebra/math/algorithms/matrix/decomposition/partial_pivot_lud.hpp"
   "include/algebra/math/algorithms/matrix/determinant/cofactor.hpp"
   "include/algebra/math/algorithms/matrix/determinant/hard_coded.hpp"
   "include/algebra/math/algorithms/matrix/determinant/partial_pivot_lud.hpp"
   "include/algebra/math/algorithms/matrix/inverse/cofactor.hpp"
   "include/algebra/math/algorithms/matrix/inverse/hard_coded.hpp"
   "include/algebra/math/algorithms/matrix/inverse/partial_pivot_lud.hpp"
   "include/algebra/math/algorithms/utils/algorithm_finder.hpp")
target_link_libraries(algebra_cmath_math
   INTERFACE algebra::common algebra::common_math)
algebra_test_public_headers( algebra_cmath_math
   "algebra/math/cmath.hpp" )
