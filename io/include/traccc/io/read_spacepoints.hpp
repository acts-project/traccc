/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/geometry.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>
#include <string_view>

namespace traccc::io {

/// Read cell data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param event The event ID to read in the cells for
/// @param directory The directory holding the cell data files
/// @param geom The description of the detector geometry
/// @param format The format of the cell data files (to read)
/// @param mr The memory resource to create the host container with
/// @return A cell (host) container
///
spacepoint_container_types::host read_spacepoints(
    std::size_t event, std::string_view directory, const geometry &geom,
    data_format format = data_format::csv,
    vecmem::memory_resource *mr = nullptr);

/// Read cell data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param filename The file to read the cell data from
/// @param geom The description of the detector geometry
/// @param format The format of the cell data files (to read)
/// @param mr The memory resource to create the host container with
/// @return A cell (host) container
///
spacepoint_container_types::host read_spacepoints(
    std::string_view filename, const geometry &geom,
    data_format format = data_format::csv,
    vecmem::memory_resource *mr = nullptr);

}  // namespace traccc::io
