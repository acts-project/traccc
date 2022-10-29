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
#include "traccc/io/binary.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/io/utils.hpp"

namespace traccc {

/// Function for cell file writing
///
/// @param event is the event index
/// @param cells_directory is the directory of cell file
/// @param data_format is the data format (e.g. csv or binary) of output file
/// @param cells_per_event is input cell container
inline void write_cells(
    size_t event, const std::string &cells_directory,
    const traccc::data_format &data_format,
    const traccc::cell_container_types::host &cells_per_event) {

    if (data_format == traccc::data_format::binary) {

        std::string io_cells_file = data_directory() + cells_directory +
                                    get_event_filename(event, "-cells.dat");

        traccc::write_binary(io_cells_file, cells_per_event);
    } else {
        throw std::invalid_argument("Allowed data format is binary");
    }
}

/// Function for hit file writing
///
/// @param event is the event index
/// @param hits_directory is the directory of hit file
/// @param data_format is the data format (e.g. csv or binary) of output file
/// @param spacepoints_per_event is input spacepoint container
inline void write_spacepoints(
    size_t event, const std::string &hits_directory,
    const traccc::data_format &data_format,
    const spacepoint_container_types::host &spacepoints_per_event) {

    if (data_format == traccc::data_format::binary) {

        std::string io_hits_file = data_directory() + hits_directory +
                                   get_event_filename(event, "-hits.dat");

        traccc::write_binary(io_hits_file, spacepoints_per_event);
    } else {
        throw std::invalid_argument("Allowed data format is binary");
    }
}

/// Function for measurement file writing
///
/// @param event is the event index
/// @param measurements_directory is the directory of hit file
/// @param data_format is the data format (e.g. csv or binary) of output file
/// @param measurements_per_event is input measurement container
inline void write_measurements(
    size_t event, const std::string &measurements_directory,
    const traccc::data_format &data_format,
    const measurement_container_types::host &measurements_per_event) {

    if (data_format == traccc::data_format::binary) {

        std::string io_measurements_file =
            data_directory() + measurements_directory +
            get_event_filename(event, "-measurements.dat");

        traccc::write_binary(io_measurements_file, measurements_per_event);
    } else {
        throw std::invalid_argument("Allowed data format is binary");
    }
}

}  // namespace traccc
