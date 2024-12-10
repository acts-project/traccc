/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/utils/ranges.hpp"

namespace detray::detail {

/// Generate phi values along an arc
///
/// @param start_phi is the start for the arc generation
/// @param end_phi is the end of the arc generation
/// @param n_seg is the number of segments used to gnerate the arc
///
/// @return a vector of phi values for the arc
template <typename scalar_t>
static inline dvector<scalar_t> phi_values(scalar_t start_phi, scalar_t end_phi,
                                           dindex n_seg) {
    dvector<scalar_t> values;
    values.reserve(n_seg + 1u);
    scalar_t step_phi = (end_phi - start_phi) / static_cast<scalar_t>(n_seg);
    for (unsigned int istep = 0u; istep <= n_seg; ++istep) {
        values.push_back(start_phi + static_cast<scalar_t>(istep) * step_phi);
    }
    return values;
}

/// Create a r-phi polygon from principle parameters
///
/// @param rmin minum r parameter
/// @param rmax maximum r parameter
/// @param phimin minimum phi parameter
/// @param phimax maximum phi parameters
///
/// @return a polygon representation of the bin
template <typename scalar_t, typename point2_t>
inline std::vector<point2_t> r_phi_polygon(scalar_t rmin, scalar_t rmax,
                                           scalar_t phimin, scalar_t phimax,
                                           unsigned int n_segments = 1u) {

    std::vector<point2_t> r_phi_poly;
    r_phi_poly.reserve(2u * n_segments + 2u);

    scalar_t cos_min_phi = math::cos(phimin);
    scalar_t sin_min_phi = math::sin(phimin);
    scalar_t cos_max_phi = math::cos(phimax);
    scalar_t sin_max_phi = math::sin(phimax);

    // @TODO add phi generators
    r_phi_poly.push_back({rmin * cos_min_phi, rmin * sin_min_phi});
    r_phi_poly.push_back({rmin * cos_max_phi, rmin * sin_max_phi});
    r_phi_poly.push_back({rmax * cos_max_phi, rmax * sin_max_phi});
    r_phi_poly.push_back({rmax * cos_min_phi, rmax * sin_min_phi});

    return r_phi_poly;
}

/// Functor to produce vertices on a mask collection in a mask tuple container.
template <typename point2_t, typename point3_t>
struct vertexer {

    /// Specialized method to generate vertices per maks group
    ///
    /// @tparam mask_group_t is the type of the mask collection in a mask cont.
    /// @tparam mask_range_t is the type of the according mask range object
    ///
    /// @param masks is the associated (and split out) mask group
    /// @param range is the range list of masks to be processed
    ///
    /// @return a jagged vector of points of the mask vertices (one per maks)
    template <typename mask_group_t, typename mask_range_t>
    dvector<dvector<point3_t>> operator()(const mask_group_t &masks,
                                          const mask_range_t &range,
                                          unsigned int n_segments = 1) {
        dvector<dvector<point3_t>> mask_vertices = {};
        for (auto i : detray::views::iota(range)) {
            mask_vertices.push_back(masks[i].vertices(n_segments));
        }
        return mask_vertices;
    }
};

}  // namespace detray::detail
