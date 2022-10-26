/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/singlet.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc {

inline std::pair<detray::axis::circular<>, detray::axis::regular<>> get_axes(
    const spacepoint_grid_config& grid_config, vecmem::memory_resource& mr) {

    // calculate circle intersections of helix and max detector radius
    scalar minHelixRadius =
        grid_config.minPt / (300. * grid_config.bFieldInZ);  // in mm
    scalar maxR2 = grid_config.rMax * grid_config.rMax;
    scalar xOuter = maxR2 / (2 * minHelixRadius);
    scalar yOuter = std::sqrt(maxR2 - xOuter * xOuter);
    scalar outerAngle = std::atan(xOuter / yOuter);
    // intersection of helix and max detector radius minus maximum R distance
    // from middle SP to top SP
    scalar innerAngle = 0;
    if (grid_config.rMax > grid_config.deltaRMax) {
        scalar innerCircleR2 = (grid_config.rMax - grid_config.deltaRMax) *
                               (grid_config.rMax - grid_config.deltaRMax);
        scalar xInner = innerCircleR2 / (2 * minHelixRadius);
        scalar yInner = std::sqrt(innerCircleR2 - xInner * xInner);
        innerAngle = std::atan(xInner / yInner);
    }

    // FIXME: phibin size must include max impact parameters
    // divide 2pi by angle delta to get number of phi-bins
    // size is always 2pi even for regions of interest
    detray::dindex phiBins = std::floor(2 * M_PI / (outerAngle - innerAngle));

    detray::axis::circular m_phi_axis{phiBins, -M_PI, M_PI, mr};

    // TODO: can probably be optimized using smaller z bins
    // and returning (multiple) neighbors only in one z-direction for forward
    // seeds
    // FIXME: zBinSize must include scattering

    scalar zBinSize = grid_config.cotThetaMax * grid_config.deltaRMax;
    detray::dindex zBins =
        std::floor((grid_config.zMax - grid_config.zMin) / zBinSize);

    detray::axis::regular m_z_axis{zBins, grid_config.zMin, grid_config.zMax,
                                   mr};

    return {m_phi_axis, m_z_axis};
}

inline TRACCC_HOST_DEVICE size_t is_valid_sp(const seedfinder_config& config,
                                             const spacepoint& sp) {
    if (sp.z() > config.zMax || sp.z() < config.zMin) {
        return detray::detail::invalid_value<size_t>();
    }
    scalar spPhi = algebra::math::atan2(sp.y(), sp.x());
    if (spPhi > config.phiMax || spPhi < config.phiMin) {
        return detray::detail::invalid_value<size_t>();
    }
    size_t r_index = getter::perp(
        vector2{sp.x() - config.beamPos[0], sp.y() - config.beamPos[1]});

    if (r_index < config.get_num_rbins()) {
        return r_index;
    }

    return detray::detail::invalid_value<size_t>();
}

template <typename spacepoint_container_t,
          template <typename> class jagged_vector_type>
inline TRACCC_HOST_DEVICE void fill_radius_bins(
    const seedfinder_config& config, const spacepoint_container_t& sp_container,
    const sp_location sp_loc, jagged_vector_type<sp_location>& r_bins) {

    const spacepoint& sp =
        sp_container.get_items()[sp_loc.bin_idx][sp_loc.sp_idx];

    auto r_index = is_valid_sp(config, sp);

    if (r_index != detray::detail::invalid_value<size_t>()) {
        r_bins[r_index].push_back(sp_loc);
    }
}

}  // namespace traccc
