/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include "detray/geometry/surface.hpp"

namespace traccc::device {

template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, typename detector_t::view_type det_data,
    const int n_params,
    bound_track_parameters_collection_types::view params_view) {

    // Type definitions
    using algebra_type = typename detector_t::algebra_type;
    using interactor_type = detray::pointwise_material_interactor<algebra_type>;

    // Detector
    detector_t det(det_data);

    // in param
    bound_track_parameters_collection_types::device params(params_view);

    if (globalIndex >= n_params) {
        return;
    }

    auto& bound_param = params.at(globalIndex);

    // Get intersection at surface
    const detray::surface sf{det, bound_param.surface_link()};
    const typename detector_t::geometry_context ctx{};

    // Apply interactor
    typename interactor_type::state interactor_state;
    interactor_type{}.update(
        bound_param, interactor_state,
        static_cast<int>(detray::navigation::direction::e_forward), sf,
        std::abs(
            sf.cos_angle(ctx, bound_param.dir(), bound_param.bound_local())));
}

}  // namespace traccc::device
