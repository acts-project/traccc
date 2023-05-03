/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_cells.hpp"

#include "csv/read_cells.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

void read_cells(cell_reader_output& out, std::size_t event,
                std::string_view directory, data_format format,
                const geometry* geom, const digitization_config* dconfig) {

    switch (format) {
        case data_format::csv: {
            read_cells(out,
                       data_directory() + directory.data() +
                           get_event_filename(event, "-cells.csv"),
                       format, geom, dconfig);
            break;
        }
        case data_format::binary: {
            details::read_binary_collection<cell_collection_types::host>(
                out.cells, data_directory() + directory.data() +
                               get_event_filename(event, "-cells.dat"));
            details::read_binary_collection<cell_module_collection_types::host>(
                out.modules, data_directory() + directory.data() +
                                 get_event_filename(event, "-modules.dat"));
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_cells(cell_reader_output& out, std::string_view filename,
                data_format format, const geometry* geom,
                const digitization_config* dconfig) {

    switch (format) {
        case data_format::csv:
            return csv::read_cells(out, filename, geom, dconfig);

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
