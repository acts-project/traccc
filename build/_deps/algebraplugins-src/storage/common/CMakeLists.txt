# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library(algebra_common_storage common_storage
   "include/algebra/storage/array_operators.hpp"
   "include/algebra/storage/vector.hpp")
target_link_libraries(algebra_common_storage
   INTERFACE algebra::common)
