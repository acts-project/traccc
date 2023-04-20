/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_particles.hpp"

#include "csv/read_particles.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

particle_collection_types::host read_particles(std::size_t event,
                                               std::string_view directory,
                                               data_format format,
                                               vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return read_particles(
                data_directory() + directory.data() +
                    get_event_filename(event, "-particles.csv"),
                format, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

particle_collection_types::host read_particles(std::string_view filename,
                                               data_format format,
                                               vecmem::memory_resource* mr) {

    switch (format) {
        case data_format::csv:
            return csv::read_particles(filename, mr);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
