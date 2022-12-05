/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/spacepoint_binning_helper.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_grid_capacities(
    std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid::axis_p0_type& phi_axis, const sp_grid::axis_p1_type& z_axis,
    const spacepoint_container_types::const_view& spacepoints_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        sp_prefix_sum_view,
    vecmem::data::vector_view<unsigned int> grid_capacities_view) {

    // Check if anything needs to be done.
    vecmem::device_vector<const prefix_sum_element_t> sp_prefix_sum(
        sp_prefix_sum_view);
    if (globalIndex >= sp_prefix_sum.size()) {
        return;
    }

    // Get the spacepoint that we need to look at.
    const prefix_sum_element_t sp_idx = sp_prefix_sum[globalIndex];
    const spacepoint_container_types::const_device spacepoints(
        spacepoints_view);
    const spacepoint sp = spacepoints.at(sp_idx);

    /// Check out if the spacepoint can be used for seeding.
    if (is_valid_sp(config, sp) != detray::detail::invalid_value<size_t>()) {

        // Find the grid bin that the spacepoint belongs to.
        const internal_spacepoint<spacepoint> isp(
            spacepoints, {sp_idx.first, sp_idx.second}, config.beamPos);
        const std::size_t bin_index =
            phi_axis.bin(isp.phi()) + phi_axis.bins() * z_axis.bin(isp.z());

        // Increase the capacity of the grid bin.
        vecmem::device_vector<unsigned int> grid_capacities(
            grid_capacities_view);
        vecmem::device_atomic_ref<unsigned int> bin_content(
            grid_capacities[bin_index]);
        bin_content.fetch_add(1);
    }
}

}  // namespace traccc::device
