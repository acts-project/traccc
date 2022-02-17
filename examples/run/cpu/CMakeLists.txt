# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable( seq_example "seq_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::algorithms traccc::io Boost::program_options)

traccc_add_executable( ccl_example "ccl_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::algorithms traccc::io)
