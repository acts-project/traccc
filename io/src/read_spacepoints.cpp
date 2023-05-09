/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_spacepoints.hpp"

#include "csv/read_spacepoints.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

void read_spacepoints(spacepoint_reader_output& out, std::size_t event,
                      std::string_view directory, const geometry& geom,
                      data_format format) {

    switch (format) {
        case data_format::csv: {
            read_spacepoints(out,
                             data_directory() + directory.data() +
                                 get_event_filename(event, "-hits.csv"),
                             geom, format);
            break;
        }
        case data_format::binary: {
            details::read_binary_collection<spacepoint_collection_types::host>(
                out.spacepoints, data_directory() + directory.data() +
                                     get_event_filename(event, "-hits.dat"));
            details::read_binary_collection<cell_module_collection_types::host>(
                out.modules, data_directory() + directory.data() +
                                 get_event_filename(event, "-modules.dat"));
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_spacepoints(spacepoint_reader_output& out, std::string_view filename,
                      const geometry& geom, data_format format) {

    switch (format) {
        case data_format::csv:
            return csv::read_spacepoints(out, filename, geom);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
