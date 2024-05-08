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
/// @param mr The memory resource to create the host collection with
/// @return A particle (host) collection
///
void read_particles(particle_collection_types::host& particles,
                    std::string_view filename);

}  // namespace traccc::io::csv
