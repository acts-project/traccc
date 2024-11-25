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

void read_cells(edm::silicon_cell_collection::host& cells, std::size_t event,
                std::string_view directory,
                const silicon_detector_description::host* dd,
                data_format format, bool deduplicate,
                bool use_acts_geometry_id) {

    switch (format) {
        case data_format::csv:
            read_cells(
                cells,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.csv")))
                                      .native()),
                dd, format, deduplicate, use_acts_geometry_id);
            break;

        case data_format::binary:
            read_cells(
                cells,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(
                                       get_event_filename(event, "-cells.dat")))
                                      .native()),
                dd, format, deduplicate);
            break;

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_cells(edm::silicon_cell_collection::host& cells,
                std::string_view filename,
                const silicon_detector_description::host* dd,
                data_format format, bool deduplicate,
                bool use_acts_geometry_id) {

    switch (format) {
        case data_format::csv:
            csv::read_cells(cells, filename, dd, deduplicate,
                            use_acts_geometry_id);
            break;

        case data_format::binary:
            details::read_binary_soa(cells, filename);
            break;

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
