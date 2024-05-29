/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/particle.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read (basic) particle information from a specific CSV file
///
/// @param particles A particle collection to fill
/// @param filename The file to read the particle data from
///
void read_particles(particle_collection_types::host& particles,
                    std::string_view filename);

/// Read full truth particle data into memory
///
/// @param particles A particle container to fill
/// @param particles_file The file to read the particle data from
/// @param hits_file The file to read the simulated hits from
/// @param measurements_file The file to read the "Acts measurements" from
/// @param hit_map_file The file to read the hit->measurement mapping from
/// @param bardoce_map An object to perform barcode re-mapping with
///                    (For Acts->Detray identifier re-mapping, if necessary)
///
void read_particles(particle_container_types::host& particles,
                    std::string_view particles_file, std::string_view hits_file,
                    std::string_view measurements_file,
                    std::string_view hit_map_file,
                    const std::map<std::uint64_t, detray::geometry::barcode>*
                        barcode_map = nullptr);

}  // namespace traccc::io::csv
