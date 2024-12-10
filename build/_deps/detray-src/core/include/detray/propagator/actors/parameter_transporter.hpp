/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/track_parametrization.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/detail/jacobian_engine.hpp"

namespace detray {

template <typename algebra_t>
struct parameter_transporter : actor {

    /// @name Type definitions for the struct
    /// @{
    using scalar_type = dscalar<algebra_t>;
    // Transformation matching this struct
    using transform3_type = dtransform3D<algebra_t>;
    // Matrix actor
    using matrix_operator = dmatrix_operator<algebra_t>;
    // bound matrix type
    using bound_matrix_t = bound_matrix<algebra_t>;
    // Matrix type for bound to free jacobian
    using bound_to_free_matrix_t = bound_to_free_matrix<algebra_t>;
    /// @}

    struct get_full_jacobian_kernel {

        template <typename mask_group_t, typename index_t,
                  typename stepper_state_t>
        DETRAY_HOST_DEVICE inline bound_matrix_t operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3,
            const bound_to_free_matrix_t& bound_to_free_jacobian,
            const material<scalar_type>* vol_mat_ptr,
            const stepper_state_t& stepping) const {

            using frame_t = typename mask_group_t::value_type::shape::
                template local_frame_type<algebra_t>;

            using jacobian_engine_t = detail::jacobian_engine<frame_t>;

            using free_matrix_t = free_matrix<algebra_t>;
            using free_to_bound_matrix_t =
                typename jacobian_engine_t::free_to_bound_matrix_type;

            // Free to bound jacobian at the destination surface
            const free_to_bound_matrix_t free_to_bound_jacobian =
                jacobian_engine_t::free_to_bound_jacobian(trf3, stepping());

            // Path correction factor
            const free_matrix_t path_correction =
                jacobian_engine_t::path_correction(
                    stepping().pos(), stepping().dir(), stepping.dtds(),
                    stepping.dqopds(vol_mat_ptr), trf3);

            const free_matrix_t correction_term =
                matrix_operator()
                    .template identity<e_free_size, e_free_size>() +
                path_correction;

            return free_to_bound_jacobian * correction_term *
                   stepping.transport_jacobian() * bound_to_free_jacobian;
        }
    };

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(propagator_state_t& propagation) const {
        auto& stepping = propagation._stepping;
        const auto& navigation = propagation._navigation;

        // Do covariance transport when the track is on surface
        if (!(navigation.is_on_sensitive() ||
              navigation.encountered_sf_material())) {
            return;
        }

        // Geometry context for this track
        const auto& gctx = propagation._context;

        // Current Surface
        const auto sf = navigation.get_surface();

        // Bound track params of departure surface
        auto& bound_params = stepping.bound_params();

        // Covariance is transported only when the previous surface is an
        // actual tracking surface. (i.e. This disables the covariance transport
        // from curvilinear frame)
        if (!bound_params.surface_link().is_invalid()) {

            // Previous surface
            tracking_surface prev_sf{navigation.detector(),
                                     bound_params.surface_link()};

            const bound_to_free_matrix_t bound_to_free_jacobian =
                prev_sf.bound_to_free_jacobian(gctx, bound_params);

            auto vol = navigation.get_volume();
            const auto vol_mat_ptr =
                vol.has_material() ? vol.material_parameters(stepping().pos())
                                   : nullptr;
            stepping.set_full_jacobian(
                sf.template visit_mask<get_full_jacobian_kernel>(
                    sf.transform(gctx), bound_to_free_jacobian, vol_mat_ptr,
                    propagation._stepping));

            // Calculate surface-to-surface covariance transport
            const bound_matrix_t new_cov =
                stepping.full_jacobian() * bound_params.covariance() *
                matrix_operator().transpose(stepping.full_jacobian());

            stepping.bound_params().set_covariance(new_cov);
        }

        // Convert free to bound vector
        bound_params.set_parameter_vector(
            sf.free_to_bound_vector(gctx, stepping()));

        // Set surface link
        bound_params.set_surface_link(sf.barcode());
    }

};  // namespace detray

}  // namespace detray
