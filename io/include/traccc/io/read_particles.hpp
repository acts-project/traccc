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
/// @param mr The memory resource to create the host collection with
/// @return A particle (host) collection
///
void read_particles(particle_collection_types::host &particles,
                    std::size_t event, std::string_view directory,
                    data_format format = data_format::csv);

/// Read basic truth particle data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param particles A particle collection to fill
/// @param filename The file to read the particle data from
/// @param format The format of the particle data files (to read)
/// @param mr The memory resource to create the host collection with
/// @return A particle (host) collection
///
void read_particles(particle_collection_types::host &particles,
                    std::string_view filename,
                    data_format format = data_format::csv);

}  // namespace traccc::io
