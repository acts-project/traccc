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
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function used for calculating the number of spacepoint doublets
///
/// The count is necessary for allocating the appropriate amount of memory
/// for storing the information of the candidates in a next step.
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] config        Seedfinder configuration
/// @param[in] sp_view       The spacepoint grid to count doublets on
/// @param[in] sp_ps_view    Prefix sum for iterating over the spacepoint grid
/// @param[out] doublet_view Container storing the number of doublets
///
TRACCC_HOST_DEVICE
inline void count_doublets(
    std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid_const_view& sp_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& sp_ps_view,
    doublet_counter_container_types::view doublet_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/count_doublets.ipp"
