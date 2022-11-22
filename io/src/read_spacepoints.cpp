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

spacepoint_container_types::host read_spacepoints(std::size_t event,
                                                  std::string_view directory,
                                                  const geometry& geom,
                                                  data_format format,
                                                  vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return read_spacepoints(data_directory() + directory.data() +
                                        get_event_filename(event, "-hits.csv"),
                                    geom, format, mr);
        case data_format::binary:
            return read_spacepoints(data_directory() + directory.data() +
                                        get_event_filename(event, "-hits.dat"),
                                    geom, format, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

spacepoint_container_types::host read_spacepoints(std::string_view filename,
                                                  const geometry& geom,
                                                  data_format format,
                                                  vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return csv::read_spacepoints(filename, geom, mr);
        case data_format::binary:
            return details::read_binary_container<
                spacepoint_container_types::host>(filename, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
