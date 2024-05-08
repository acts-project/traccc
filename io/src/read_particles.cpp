/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_particles.hpp"

#include "csv/read_particles.hpp"
#include "traccc/io/utils.hpp"

// System include(s).
#include <filesystem>

namespace traccc::io {

void read_particles(particle_collection_types::host& particles,
                    std::size_t event, std::string_view directory,
                    data_format format) {

    switch (format) {
        case data_format::csv:
            read_particles(
                particles,
                get_absolute_path((std::filesystem::path(directory) /
                                   std::filesystem::path(get_event_filename(
                                       event, "-particles.csv")))
                                      .native()),
                format);
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_particles(particle_collection_types::host& particles,
                    std::string_view filename, data_format format) {

    switch (format) {
        case data_format::csv:
            csv::read_particles(particles, filename);
            break;
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
