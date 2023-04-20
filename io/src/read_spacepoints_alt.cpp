/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_spacepoints_alt.hpp"

#include "csv/read_spacepoints_alt.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

spacepoint_reader_output read_spacepoints_alt(std::size_t event,
                                              std::string_view directory,
                                              const geometry& geom,
                                              data_format format,
                                              vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return read_spacepoints_alt(
                data_directory() + directory.data() +
                    get_event_filename(event, "-hits.csv"),
                geom, format, mr);
        case data_format::binary: {
            auto hits = details::read_binary_collection<
                spacepoint_collection_types::host>(
                data_directory() + directory.data() +
                    get_event_filename(event, "-hits.dat"),
                mr);
            auto modules = details::read_binary_collection<
                cell_module_collection_types::host>(
                data_directory() + directory.data() +
                    get_event_filename(event, "-modules.dat"),
                mr);
            return {std::move(hits), std::move(modules)};
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

spacepoint_reader_output read_spacepoints_alt(std::string_view filename,
                                              const geometry& geom,
                                              data_format format,
                                              vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return csv::read_spacepoints_alt(filename, geom, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
