/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/geometry.hpp"
#include "traccc/io/reader_edm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read spacepoint information from a specific CSV file
///
/// @param filename The file to read the spacepoint data from
/// @param geom The description of the detector geometry
/// @param mr The memory resource to create the host collection with
/// @return A spacepoint (host) collection
///
spacepoint_reader_output read_spacepoints_alt(
    std::string_view filename, const geometry& geom,
    vecmem::memory_resource* mr = nullptr);

}  // namespace traccc::io::csv
