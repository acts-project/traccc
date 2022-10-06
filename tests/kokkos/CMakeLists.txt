# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_test( kokkos
   kokkos_main.cpp
   kokkos_basic.cpp
   LINK_LIBRARIES
   GTest::gtest
   Kokkos::kokkos
)
