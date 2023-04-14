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
    const spacepoint_collection_types::host& sp_collection) const {

    output_type g2(m_axes.first, m_axes.second, m_mr.get());

    auto& phi_axis = g2.axis_p0();
    auto& z_axis = g2.axis_p1();

    for (unsigned int i = 0; i < sp_collection.size(); i++) {
        const spacepoint& sp = sp_collection[i];
        internal_spacepoint<spacepoint> isp(sp, i, m_config.beamPos);

        if (is_valid_sp(m_config, sp) !=
            detray::detail::invalid_value<size_t>()) {
            const std::size_t bin_index =
                phi_axis.bin(isp.phi()) + phi_axis.bins() * z_axis.bin(isp.z());
            g2.bin(bin_index).push_back(std::move(isp));
        }
    }
    return g2;
}

}  // namespace traccc
