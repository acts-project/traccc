# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable( par_example "par_example.cpp"
   LINK_LIBRARIES OpenMP::OpenMP_CXX vecmem::core traccc::core traccc::io
                  traccc::options )
