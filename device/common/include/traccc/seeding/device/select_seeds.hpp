/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

// System include(s).
#include <cassert>
#include <cstddef>

namespace traccc::device {

/// Function used for selecting good triplets to be recorded into seed container
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] filter_config seed filter config
/// @param[in] spacepoints_view vecmem container for internal spacepoint
/// @param[in] dc_prefix_sum view of the vector prefix sum on
/// doublet_counter_container
/// @param[in] doublet_counter_container vecmem container for doublet_counter
/// @param[in] tc_view view on vecmem container for triplets
/// @param[in] data Array for temporary storage of triplets for
/// comparison
/// @param[out] seed_container vecmem container for seeds
///
TRACCC_HOST_DEVICE
inline void select_seeds(
    std::size_t globalIndex, const seedfilter_config& filter_config,
    const spacepoint_container_types::const_view& spacepoints_view,
    const sp_grid_const_view& internal_sp_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& dc_ps_view,
    const device::doublet_counter_container_types::const_view&
        doublet_counter_container,
    const triplet_container_types::const_view& tc_view, triplet* data,
    seed_collection_types::view seed_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/select_seeds.ipp"