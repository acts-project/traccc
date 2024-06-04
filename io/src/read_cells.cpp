/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_cells.hpp"

#include "csv/read_cells.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

// System include(s).
#include <filesystem>

namespace traccc::io {

void read_cells(
    cell_reader_output& out, std::size_t event, std::string_view directory,
    data_format format, const geometry* geom,
    const digitization_config* dconfig,
    const std::map<std::uint64_t, detray::geometry::barcode>* barcode_map,
    bool deduplicate) {

    switch (format) {
        case data_format::csv: {
            read_cells(
                out,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.csv")))
                                      .native()),
                format, geom, dconfig, barcode_map, deduplicate);
            break;
        }
        case data_format::binary: {
            details::read_binary_collection<cell_collection_types::host>(
                out.cells,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.dat")))
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

void read_cells(
    cell_reader_output& out, std::string_view filename, data_format format,
    const geometry* geom, const digitization_config* dconfig,
    const std::map<std::uint64_t, detray::geometry::barcode>* barcode_map,
    bool deduplicate) {

    switch (format) {
        case data_format::csv:
            return csv::read_cells(out, filename, geom, dconfig, barcode_map,
                                   deduplicate);

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
