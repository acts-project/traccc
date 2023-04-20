/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// System include(s)
#include <cstddef>

namespace traccc::device {

/// Function used for updating the triplets' weights
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] filter_config Seedfilter configuration
/// @param[in] sp_view       The spacepoint grid
/// @param[in] spM_tc_view   Collection of triplet counts per spM
/// @param[in] tc_view       Collection of triplet counts per midBot doublet
/// @param[in] data Array for temporary storage of quality parameters for
/// comparison of triplets
/// @param[inout] triplet_view Collection of triplets
///
TRACCC_HOST_DEVICE
inline void update_triplet_weights(
    std::size_t globalIndex, const seedfilter_config& filter_config,
    const sp_grid_const_view& sp_view,
    const triplet_counter_spM_collection_types::const_view& spM_tc_view,
    const triplet_counter_collection_types::const_view& tc_view, scalar* data,
    device_triplet_collection_types::view triplet_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/update_triplet_weights.ipp"