/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_measurements.hpp"

#include "csv/read_measurements.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

// System include(s).
#include <filesystem>

namespace traccc::io {

void read_measurements(measurement_reader_output& out, std::size_t event,
                       std::string_view directory, data_format format) {

    switch (format) {
        case data_format::csv: {
            read_measurements(
                out,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-measurements.csv")))
                                      .native()),
                format);
            break;
        }
        case data_format::binary: {

            details::read_binary_collection<measurement_collection_types::host>(
                out.measurements,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-measurements.dat")))
                                      .native()));
            details::read_binary_collection<cell_module_collection_types::host>(
                out.modules,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-modules.dat")))
                                      .native()));
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_measurements(measurement_reader_output& out,
                       std::string_view filename, data_format format) {

    switch (format) {
        case data_format::csv:
            return csv::read_measurements(out, filename);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
