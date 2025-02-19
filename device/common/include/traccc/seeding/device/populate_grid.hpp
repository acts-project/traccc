/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

namespace traccc::device {

/// Function populating the spacepoint grid
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] config        Seedfinder configuration
/// @param[in] spacepoints   All the spacepoints of the event
/// @param[out] grid         The spacepoint grid to populate
///
TRACCC_HOST_DEVICE
inline void populate_grid(
    global_index_t globalIndex, const seedfinder_config& config,
    const spacepoint_collection_types::const_view& spacepoints,
    sp_grid_view grid);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/populate_grid.ipp"
