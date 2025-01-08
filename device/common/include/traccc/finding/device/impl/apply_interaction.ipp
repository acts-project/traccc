/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/actors/pointwise_material_interactor.hpp>

namespace traccc::device {

template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    const global_index_t globalIndex, const finding_config& cfg,
    const apply_interaction_payload<detector_t>& payload) {

    // Type definitions
    using algebra_type = typename detector_t::algebra_type;
    using interactor_type = detray::pointwise_material_interactor<algebra_type>;

    // Detector
    detector_t det(payload.det_data);

    // in param
    bound_track_parameters_collection_types::device params(payload.params_view);
    vecmem::device_vector<const unsigned int> params_liveness(
        payload.params_liveness_view);

    if (globalIndex >= payload.n_params) {
        return;
    }

    auto& bound_param = params.at(globalIndex);

    if (params_liveness.at(globalIndex) != 0u) {
        // Get surface corresponding to bound params
        const detray::tracking_surface sf{det, bound_param.surface_link()};
        const typename detector_t::geometry_context ctx{};

        // Apply interactor
        typename interactor_type::state interactor_state;
        interactor_type{}.update(
            ctx,
            detail::correct_particle_hypothesis(cfg.ptc_hypothesis,
                                                bound_param),
            bound_param, interactor_state,
            static_cast<int>(detray::navigation::direction::e_forward), sf);
    }
}
}  // namespace traccc::device
