# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the library.
algebra_add_library( algebra_vc_storage vc_storage
   "include/algebra/storage/vc.hpp"
   "include/algebra/storage/impl/vc_array4_wrapper.hpp" )
target_link_libraries( algebra_vc_storage
   INTERFACE algebra::common Vc::Vc )
algebra_test_public_headers( algebra_vc_storage
   "algebra/storage/vc.hpp" )
