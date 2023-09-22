/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include "detray/intersection/detail/trajectories.hpp"
#include "detray/intersection/intersection_kernel.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/navigator.hpp"

namespace traccc::device {

template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, typename detector_t::detector_view_type det_data,
    vecmem::data::jagged_vector_view<detray::intersection2D<
        typename detector_t::surface_type, typename detector_t::transform3>>
        nav_candidates_buffer,
    const int n_params,
    bound_track_parameters_collection_types::view params_view) {

    // Type definitions
    using transform3_type = typename detector_t::transform3;
    using intersection_type =
        detray::intersection2D<typename detector_t::surface_type,
                               transform3_type>;
    using interactor_type =
        detray::pointwise_material_interactor<transform3_type>;

    // Detector
    detector_t det(det_data);

    // Navigation candidate buffer
    vecmem::jagged_device_vector<intersection_type> nav_candidates(
        nav_candidates_buffer);

    // in param
    bound_track_parameters_collection_types::device params(params_view);

    if (static_cast<int>(globalIndex) >= n_params) {
        return;
    }

    auto& bound_param = params.at(globalIndex);

    // Get intersection at surface
    const detray::surface<detector_t> sf{det, bound_param.surface_link()};
    using cxt_t = typename detector_t::geometry_context;
    const cxt_t ctx{};
    const auto free_vec = sf.bound_to_free_vector(ctx, bound_param.vector());

    intersection_type sfi;
    sfi.sf_desc = det.surface(bound_param.surface_link());
    sf.template visit_mask<detray::intersection_update>(
        detray::detail::ray<transform3_type>(free_vec), sfi,
        det.transform_store());

    // Apply interactor
    typename interactor_type::state interactor_state;
    interactor_type{}.update(
        bound_param, interactor_state,
        static_cast<int>(detray::navigation::direction::e_forward), sf,
        sfi.cos_incidence_angle);
}

}  // namespace traccc::device
