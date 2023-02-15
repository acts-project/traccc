/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/spacepoint_binning_helper.hpp"

namespace traccc::device {

TRACCC_DEVICE
inline void populate_grid(
    unsigned int globalIndex, const seedfinder_config& config,
    const spacepoint_collection_types::const_view& spacepoints_view,
    sp_grid_view grid_view) {

    // Check if anything needs to be done.
    const spacepoint_collection_types::const_device spacepoints(
        spacepoints_view);
    if (globalIndex >= spacepoints.size()) {
        return;
    }
    const spacepoint sp = spacepoints.at(globalIndex);

    /// Check out if the spacepoint can be used for seeding.
    if (is_valid_sp(config, sp) != detray::detail::invalid_value<size_t>()) {

        // Set up the spacepoint grid object(s).
        sp_grid_device grid(grid_view);
        const sp_grid_device::axis_p0_type& phi_axis = grid.axis_p0();
        const sp_grid_device::axis_p1_type& z_axis = grid.axis_p1();

        // Find the grid bin that the spacepoint belongs to.
        const internal_spacepoint<spacepoint> isp(sp, globalIndex,
                                                  config.beamPos);
        const std::size_t bin_index =
            phi_axis.bin(isp.phi()) + phi_axis.bins() * z_axis.bin(isp.z());

        // Add the spacepoint to the grid.
        grid.bin(bin_index).push_back(std::move(isp));
    }
}

}  // namespace traccc::device
