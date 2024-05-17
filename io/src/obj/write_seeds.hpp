/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::obj {

/// Write a seed collection into a Wavefront OBJ file.
///
/// @param filename is the name of the output file
/// @param seeds is the seed collection to write
/// @param spacepoints is the spacepoint collection that the seeds reference
///
void write_seeds(std::string_view filename,
                 seed_collection_types::const_view seeds,
                 spacepoint_collection_types::const_view spacepoints);

}  // namespace traccc::io::obj
