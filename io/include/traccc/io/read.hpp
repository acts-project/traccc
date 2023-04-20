/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"
#include "traccc/io/demonstrator_alt_edm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <string_view>

namespace traccc::io {

/// Read input data for a specified number of events
///
/// This function can read data about multiple events into memory at the
/// same time. Even using (OpenMP) parallelism for the reading if possible.
///
/// @param events The number of events to read input data for
/// @param directory The directory to read the cell data from
/// @param detector_file The file describing the detector geometry
/// @param digi_config_file The file describing the detector digitization
/// @param format The format of the event file(s)
/// @param mr The memory resource to allocate the container(s) with
/// @return An object with the requested events worth of input
///
alt_demonstrator_input read(std::size_t events, std::string_view directory,
                            std::string_view detector_file,
                            std::string_view digi_config_file,
                            data_format format = data_format::csv,
                            vecmem::memory_resource *mr = nullptr);

}  // namespace traccc::io
