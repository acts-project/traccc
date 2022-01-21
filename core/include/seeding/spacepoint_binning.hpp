/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include <seeding/spacepoint_binning_helper.hpp>

namespace traccc {

/// spacepoint binning
struct spacepoint_binning
    : public algorithm<sp_grid(const host_spacepoint_container&)> {

    /// Constructor for the spacepoint binning
    ///
    /// @param config is seed finder configuration parameters
    /// @param grid_config is for spacepoint grid parameter
    /// @param mr is the vecmem memory resource
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr)
        : m_config(config), m_grid_config(grid_config), m_mr(mr) {

        m_axes = get_axes(grid_config, mr);
    }

    output_type operator()(
        const host_spacepoint_container& sp_container) const override {
        output_type g2(m_axes.first, m_axes.second, m_mr.get());

        djagged_vector<sp_location> rbins(m_config.get_num_rbins());

        for (unsigned int i = 0; i < sp_container.size(); i++) {
            for (unsigned int j = 0; j < sp_container.get_items()[i].size();
                 j++) {
                sp_location sp_loc{i, j};
                fill_radius_bins<host_spacepoint_container, djagged_vector>(
                    m_config, sp_container, sp_loc, rbins);
            }
        }

        // fill rbins into grid such that each grid bin is sorted in r
        // space points with delta r < rbin size can be out of order
        for (auto& rbin : rbins) {
            for (auto& sp_loc : rbin) {
                const spacepoint& sp =
                    sp_container.get_items()[sp_loc.bin_idx][sp_loc.sp_idx];
                auto isp =
                    internal_spacepoint<spacepoint>(sp, m_config.beamPos);

                point2 sp_position = {isp.phi(), isp.z()};
                g2.populate(sp_position, std::move(isp));
            }
        }

        return g2;
    }

    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::pair<output_type::axis_p0_t, output_type::axis_p1_t> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
