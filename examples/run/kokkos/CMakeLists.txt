
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable( seeding_example_kokkos "seeding_example_kokkos.cpp"
   LINK_LIBRARIES vecmem::core traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::kokkos Kokkos::kokkos
                  traccc::options )
