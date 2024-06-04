/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/edm/particle.hpp"

// System include(s).
#include <cstddef>
#include <cstdint>
#include <map>
#include <string_view>

namespace traccc::io {

/// Read basic truth particle data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param particles A particle collection to fill
/// @param event The event ID to read in the cells for
/// @param directory The directory holding the particle data files
/// @param format The format of the particle data files (to read)
/// @param filename_postfix Postfix for the particle file name(s)
///
void read_particles(particle_collection_types::host &particles,
                    std::size_t event, std::string_view directory,
                    data_format format = data_format::csv,
                    std::string_view filename_postfix = "-particles_initial");

/// Read basic truth particle data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param particles A particle collection to fill
/// @param filename The file to read the particle data from
/// @param format The format of the particle data files (to read)
///
void read_particles(particle_collection_types::host &particles,
                    std::string_view filename,
                    data_format format = data_format::csv);

/// Read full truth particle data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param particles A particle container to fill
/// @param event The event ID to read in the cells for
/// @param directory The directory holding the particle data files
/// @param format The format of the particle data files (to read)
/// @param bardoce_map An object to perform barcode re-mapping with
///                    (For Acts->Detray identifier re-mapping, if necessary)
/// @param filename_postfix Postfix for the particle file name(s)
///
void read_particles(particle_container_types::host &particles,
                    std::size_t event, std::string_view directory,
                    data_format format = data_format::csv,
                    const std::map<std::uint64_t, detray::geometry::barcode>
                        *barcode_map = nullptr,
                    std::string_view filename_postfix = "-particles_initial");

/// Read full truth particle data into memory
///
/// The required file names are selected explicitly by the user.
///
/// @param particles A particle container to fill
/// @param particles_file The file to read the particle data from
/// @param hits_file The file to read the simulated hits from
/// @param measurements_file The file to read the "Acts measurements" from
/// @param hit_map_file The file to read the hit->measurement mapping from
/// @param format The format of the particle data files (to read)
/// @param bardoce_map An object to perform barcode re-mapping with
///                    (For Acts->Detray identifier re-mapping, if necessary)
///
void read_particles(
    particle_container_types::host &particles, std::string_view particles_file,
    std::string_view hits_file, std::string_view measurements_file,
    std::string_view hit_map_file, data_format format = data_format::csv,
    const std::map<std::uint64_t, detray::geometry::barcode> *barcode_map =
        nullptr);

}  // namespace traccc::io
