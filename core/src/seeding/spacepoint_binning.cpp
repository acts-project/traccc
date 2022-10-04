/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/spacepoint_binning.hpp"

#include "traccc/definitions/primitives.hpp"
#include "traccc/seeding/spacepoint_binning_helper.hpp"

namespace traccc {

spacepoint_binning::spacepoint_binning(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    vecmem::memory_resource& mr)
    : m_config(config.toInternalUnits()),
      m_grid_config(grid_config.toInternalUnits()),
      m_axes(get_axes(grid_config.toInternalUnits(), mr)),
      m_mr(mr) {}

spacepoint_binning::output_type spacepoint_binning::operator()(
    const spacepoint_container_types::host& sp_container) const {

    output_type g2(m_axes.first, m_axes.second, m_mr.get());

    djagged_vector<sp_location> rbins(m_config.get_num_rbins());

    for (unsigned int i = 0; i < sp_container.size(); i++) {
        for (unsigned int j = 0; j < sp_container.get_items()[i].size(); j++) {
            sp_location sp_loc{i, j};
            fill_radius_bins<spacepoint_container_types::host, djagged_vector>(
                m_config, sp_container, sp_loc, rbins);
        }
    }

    // fill rbins into grid such that each grid bin is sorted in r
    // space points with delta r < rbin size can be out of order
    for (auto& rbin : rbins) {
        for (auto& sp_loc : rbin) {

            auto isp = internal_spacepoint<spacepoint>(
                sp_container, {sp_loc.bin_idx, sp_loc.sp_idx},
                m_config.beamPos);

            point2 sp_position = {isp.phi(), isp.z()};
            g2.populate(sp_position, std::move(isp));
        }
    }

    return g2;
}

}  // namespace traccc
