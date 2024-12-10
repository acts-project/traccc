/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/plugins/svgtools/utils/surface_kernels.hpp"
#include "detray/utils/invalid_values.hpp"

// System include(s)
#include <cassert>
#include <tuple>

namespace detray::svgtools::utils {

/// @brief Checks if the detray surface has a volume link.
template <typename detector_t>
inline auto is_not_world_portal(
    const detray::tracking_surface<detector_t>& d_portal) {
    const auto d_link_idx = d_portal.template visit_mask<link_getter>();
    return !detail::is_invalid_value(d_link_idx);
}

/// @note expects that the detray surface has a volume link.
/// @returns the volume link of the detray surface.
template <typename detector_t>
inline auto get_linked_volume(
    const detector_t& detector,
    const detray::tracking_surface<detector_t>& d_portal) {
    assert(is_not_world_portal(d_portal));
    const auto d_link_idx = d_portal.template visit_mask<link_getter>();
    return tracking_volume{detector, d_link_idx};
}

/// @brief Calculates the start and end point of the link.
/// @note The detray surface must have a volume link.
/// @returns (start, end).
template <typename detector_t>
inline auto link_points(const typename detector_t::geometry_context& context,
                        const detector_t& detector,
                        const detray::tracking_surface<detector_t>& d_portal,
                        typename detector_t::vector3_type dir,
                        const double link_length) {
    assert(is_not_world_portal(d_portal));

    // Calculating the start position:
    const auto start = d_portal.template visit_mask<link_start_getter>(
        d_portal.transform(context));

    // Calculating the end position:
    const auto n =
        d_portal.normal(context, d_portal.global_to_local(context, start, dir));
    const auto volume = get_linked_volume(detector, d_portal);
    const auto end = d_portal.template visit_mask<link_end_getter>(
        detector, volume, start, n, link_length);

    return std::make_tuple(start, end);
}

}  // namespace detray::svgtools::utils
