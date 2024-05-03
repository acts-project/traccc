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

void read_cells(cell_collection_types::host& cells, std::size_t event,
                std::string_view directory,
                const detector_description::host* dd, data_format format,
                bool deduplicate) {

    switch (format) {
        case data_format::csv: {
            read_cells(
                out,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.csv")))
                                      .native()),
                dd, format, deduplicate);
            break;
        }
        case data_format::binary: {
            details::read_binary_collection<cell_collection_types::host>(
                out.cells,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.dat")))
                                      .native()));
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_cells(cell_collection_types::host& cells, std::string_view filename,
                const detector_description::host* dd, data_format format,
                bool deduplicate) {

    switch (format) {
        case data_format::csv:
            return csv::read_cells(cells, filename, dd, deduplicate);

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
