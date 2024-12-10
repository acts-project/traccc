# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library(algebra_fastor_math fastor_math
   "include/algebra/math/fastor.hpp"
   "include/algebra/math/impl/fastor_getter.hpp"
   "include/algebra/math/impl/fastor_matrix.hpp"
   "include/algebra/math/impl/fastor_transform3.hpp"
   "include/algebra/math/impl/fastor_vector.hpp")
target_link_libraries(algebra_fastor_math
   INTERFACE Fastor::Fastor algebra::common algebra::common_math
             algebra::fastor_storage)
algebra_test_public_headers( algebra_fastor_math
   "algebra/math/fastor.hpp" )
