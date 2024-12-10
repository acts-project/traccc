# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vecmem_storage vecmem_storage
   "include/algebra/storage/vecmem.hpp" )
target_link_libraries( algebra_vecmem_storage
   INTERFACE algebra::common vecmem::core )
algebra_test_public_headers( algebra_vecmem_storage
   "algebra/storage/vecmem.hpp" )
