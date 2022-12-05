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

/// Function used for calculating the capacity for the spacepoint grid
///
/// Before filling the spacepoint grid with the spacepoints that belong to
/// each grid bin, we need to calculate how big each of those bins are going
/// to be.
///
/// This function needs to be called separately for every spacepoint of the
/// event. Which is the same number as the size of the "prefix sum" given to
/// this function.
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] config        Seedfinder configuration
/// @param[in] phi_axis      The circular &Phi axis describing the geometry
/// @param[in] z_axis        The linear Z axis describing the geometry
/// @param[in] spacepoints   All the spacepoints of the event
/// @param[in] sp_prefix_sum The prefix sum used for the spacepoints
/// @param[out] grid_capacities Capacity required for each spacepoint grid bin
///
TRACCC_HOST_DEVICE
inline void count_grid_capacities(
    std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid::axis_p0_type& phi_axis, const sp_grid::axis_p1_type& z_axis,
    const spacepoint_container_types::const_view& spacepoints,
    const vecmem::data::vector_view<const prefix_sum_element_t>& sp_prefix_sum,
    vecmem::data::vector_view<unsigned int> grid_capacities);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/count_grid_capacities.ipp"
