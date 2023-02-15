/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function populating the spacepoint grid
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] config        Seedfinder configuration
/// @param[in] spacepoints   All the spacepoints of the event
/// @param[out] grid         The spacepoint grid to populate
///
TRACCC_DEVICE
inline void populate_grid(
    unsigned int globalIndex, const seedfinder_config& config,
    const spacepoint_collection_types::const_view& spacepoints,
    sp_grid_view grid);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/populate_grid.ipp"
