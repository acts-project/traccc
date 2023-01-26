/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_cells_alt.hpp"

#include "csv/read_cells_alt.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

alt_cell_reader_output_t read_cells_alt(std::size_t event,
                                        std::string_view directory,
                                        data_format format,
                                        const geometry* geom,
                                        const digitization_config* dconfig,
                                        vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return read_cells_alt(data_directory() + directory.data() +
                                      get_event_filename(event, "-cells.csv"),
                                  format, geom, dconfig, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

alt_cell_reader_output_t read_cells_alt(std::string_view filename,
                                        data_format format,
                                        const geometry* geom,
                                        const digitization_config* dconfig,
                                        vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return csv::read_cells_alt(filename, geom, dconfig, mr);

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
