/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/navigation/intersection/helix_cylinder_intersector.hpp"
#include "detray/navigation/intersection/helix_line_intersector.hpp"
#include "detray/navigation/intersection/helix_plane_intersector.hpp"

namespace detray {

/// @brief Intersection implementation for detector surfaces using a helix
/// trajectory.
///
/// The algorithm uses the Newton-Raphson method to find an intersection on
/// the unbounded surface and then applies the mask.
///
/// @note specialized into @c helix_plane_intersector, @c helix_line_intersector
/// and @c helix_cylinder_intersector
template <typename frame_t, typename algebra_t>
struct helix_intersector_impl {};

template <typename shape_t, typename algebra_t, bool = true>
using helix_intersector = helix_intersector_impl<
    typename shape_t::template local_frame_type<algebra_t>, algebra_t>;

}  // namespace detray
