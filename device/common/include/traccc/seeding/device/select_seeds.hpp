/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

// System include(s).
#include <cassert>
#include <cstddef>

namespace traccc::device {

/// Function used for selecting good triplets to be recorded into seed
/// collection
///
/// @param[in] globalIndex      The index of the current thread
/// @param[in] filter_config    Seed filter config
/// @param[in] spacepoints_view Collection of spacepoints
/// @param[in] internal_sp_view The spacepoint grid
/// @param[in] spM_tc_view      Collection with the number of triplets per spM
/// @param[in] triplet_view     Collection of triplets
/// @param[in] data     Array for temporary storage of triplets for comparison
/// @param[out] seed_view       Collection of seeds
///
TRACCC_HOST_DEVICE
inline void select_seeds(
    std::size_t globalIndex, const seedfilter_config& filter_config,
    const spacepoint_collection_types::const_view& spacepoints_view,
    const sp_grid_const_view& internal_sp_view,
    const triplet_counter_spM_collection_types::const_view& spM_tc_view,
    const triplet_counter_collection_types::const_view& tc_view,
    const device_triplet_collection_types::const_view& triplet_view,
    triplet* data, seed_collection_types::view seed_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/select_seeds.ipp"