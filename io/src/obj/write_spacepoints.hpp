/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/spacepoint.hpp"

#include <string_view>

namespace traccc::io::obj {

/// Write a spacepoint collection into a Wavefront OBJ file.
///
/// @param filename is the name of the output file
/// @param spacepoints is the spacepoint collection to write
///
void write_spacepoints(
    std::string_view filename,
    traccc::spacepoint_collection_types::const_view spacepoints);

}  // namespace traccc::io::obj
