# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the "build" of the traccc::io library.
traccc_add_library( traccc_io io TYPE SHARED
  # Public headers
  "include/traccc/io/detail/json_digitization_config.hpp"
  "include/traccc/io/binary.hpp"
  "include/traccc/io/csv.hpp"
  "include/traccc/io/data_format.hpp"
  "include/traccc/io/demonstrator_edm.hpp"
  "include/traccc/io/mapper.hpp"
  "include/traccc/io/write.hpp"
  "include/traccc/io/utils.hpp"
  "include/traccc/io/reader.hpp"
  # Implementation
  "src/write_binary.hpp"
  "src/write.cpp" )
target_link_libraries( traccc_io
  PUBLIC dfelibs::dfelibs vecmem::core traccc::core ActsPluginJson ActsCore)
