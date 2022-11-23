/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/io/data_format.hpp"

// System include(s).
#include <cstddef>
#include <string_view>

namespace traccc::io {

/// Function for cell file writing
///
/// @param event is the event index
/// @param directory is the directory for the output cell file
/// @param format is the data format (e.g. csv or binary) of output file
/// @param cells is the cell container to write
///
void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           traccc::cell_container_types::const_view cells);

/// Function for hit file writing
///
/// @param event is the event index
/// @param directory is the directory for the output spacepoint file
/// @param format is the data format (e.g. csv or binary) of output file
/// @param spacepoints is the spacepoint container to write
///
void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           spacepoint_container_types::const_view spacepoints);

/// Function for measurement file writing
///
/// @param event is the event index
/// @param directory is the directory for the output measurement file
/// @param format is the data format (e.g. csv or binary) of output file
/// @param measurements is the measurement container to write
///
void write(std::size_t event, std::string_view directory,
           traccc::data_format format,
           measurement_container_types::const_view measurements);

}  // namespace traccc::io
