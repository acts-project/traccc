/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/plugins/svgtools/utils/link_utils.hpp"
#include "detray/plugins/svgtools/utils/surface_kernels.hpp"

// Actsvg includes(s)
#include "actsvg/proto/portal.hpp"

namespace detray::svgtools::conversion {

/// @returns The proto link calculated using the surface normal vector.
template <typename detector_t>
inline auto link(const typename detector_t::geometry_context& context,
                 const detector_t& detector,
                 const detray::tracking_surface<detector_t>& d_portal) {

    using point3_container_t = std::vector<typename detector_t::point3_type>;
    using p_link_t = typename actsvg::proto::portal<point3_container_t>::link;

    typename detector_t::vector3_type dir{};

    // Length of link arrow is currently hardcoded.
    constexpr double link_length = 4.;

    const auto [start, end] = svgtools::utils::link_points(
        context, detector, d_portal, dir, link_length);

    p_link_t p_link;
    p_link._start = start;
    p_link._end = end;

    return p_link;
}

}  // namespace detray::svgtools::conversion
