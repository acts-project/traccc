/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include "detray/geometry/surface.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/navigation/intersection/ray_intersector.hpp"
#include "detray/navigation/intersection_kernel.hpp"

namespace traccc::device {

template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, typename detector_t::view_type det_data,
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

    if (globalIndex >= n_params) {
        return;
    }

    auto& bound_param = params.at(globalIndex);

    // Get intersection at surface
    const detray::surface<detector_t> sf{det, bound_param.surface_link()};
    using cxt_t = typename detector_t::geometry_context;
    const cxt_t ctx{};
    const auto free_vec = sf.bound_to_free_vector(ctx, bound_param.vector());

    using scalar_type = typename detector_t::scalar_type;
    const auto& mask_store = det.mask_store();
    intersection_type sfi;
    sfi.sf_desc = det.surface(bound_param.surface_link());
    sf.template visit_mask<
        detray::intersection_update<detray::ray_intersector>>(
        detray::detail::ray<transform3_type>(free_vec), sfi,
        det.transform_store(),
        sf.is_portal() ? 0.f : 15.f * unit<scalar_type>::um,
        -100.f * unit<scalar_type>::um);

    if (!(std::abs(
              sf.cos_angle(ctx, bound_param.dir(), bound_param.bound_local()) -
              sfi.cos_incidence_angle) < 0.000001f)) {
        printf("Problem");
    }

    // Apply interactor
    typename interactor_type::state interactor_state;
    interactor_type{}.update(
        bound_param, interactor_state,
        static_cast<int>(detray::navigation::direction::e_forward), sf,
        sfi.cos_incidence_angle);
}

}  // namespace traccc::device
