/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"
#include "traccc/io/reader_edm.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"

// System include(s).
#include <cstddef>
#include <string_view>

namespace traccc::io {

/// Read measurement data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param out A measurement & a cell_module (host) collections
/// @param event The event ID to read in the cells for
/// @param directory The directory holding the cell data files
/// @param format The format of the cell data files (to read)
///
void read_measurements(measurement_reader_output& out, std::size_t event,
                       std::string_view directory,
                       data_format format = data_format::csv);

/// Read measurement data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param out A measurement & a cell_module (host) collections
/// @param filename The file to read the cell data from
/// @param format The format of the cell data files (to read)
///
void read_measurements(measurement_reader_output& out,
                       std::string_view filename,
                       data_format format = data_format::csv);

/// Read measurement data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param event The event ID to read in the cells for
/// @param directory The directory holding the cell data files
/// @param format The format of the cell data files (to read)
/// @param mr The memory resource to create the host container with
/// @return A cell (host) container
///
measurement_container_types::host read_measurements_container(
    std::size_t event, std::string_view directory,
    data_format format = data_format::csv,
    vecmem::memory_resource* mr = nullptr);

/// Read measurement data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param filename The file to read the cell data from
/// @param format The format of the cell data files (to read)
/// @param mr The memory resource to create the host container with
/// @return A cell (host) container
///
measurement_container_types::host read_measurements_container(
    std::string_view filename, data_format format = data_format::csv,
    vecmem::memory_resource* mr = nullptr);

}  // namespace traccc::io
