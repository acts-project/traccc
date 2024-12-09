/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc {

/// Type unrolling functor for two-filters smoother
template <typename algebra_t>
struct two_filters_smoother {

    // Type declarations
    using matrix_operator = detray::dmatrix_operator<algebra_t>;
    using size_type = detray::dsize_type<algebra_t>;
    template <size_type ROWS, size_type COLS>
    using matrix_type = detray::dmatrix<algebra_t, ROWS, COLS>;

    /// Two-filters smoother operation
    ///
    /// @param mask_group mask group that contains the mask of surface
    /// @param index mask index of surface
    /// @param trk_state track state of the surface
    /// @param bound_params bound parameter
    ///
    /// @return true if the update succeeds
    template <typename mask_group_t, typename index_t>
    TRACCC_HOST_DEVICE inline bool operator()(
        const mask_group_t& /*mask_group*/, const index_t& /*index*/,
        track_state<algebra_t>& trk_state,
        const bound_track_parameters& bound_params) const {

        using shape_type = typename mask_group_t::value_type::shape;

        const auto D = trk_state.get_measurement().meas_dim;
        assert(D == 1u || D == 2u);
        if (D == 1u) {
            return update<1u, shape_type>(trk_state, bound_params);
        } else if (D == 2u) {
            return update<2u, shape_type>(trk_state, bound_params);
        }

        return false;
    }

    template <size_type D, typename shape_t>
    TRACCC_HOST_DEVICE inline bool update(
        track_state<algebra_t>& trk_state,
        const bound_track_parameters& bound_params) const {

        assert(trk_state.filtered().surface_link() ==
               bound_params.surface_link());

        static_assert(((D == 1u) || (D == 2u)),
                      "The measurement dimension should be 1 or 2");

        const auto meas = trk_state.get_measurement();

        matrix_type<D, e_bound_size> H = meas.subs.template projector<D>();

        // Measurement data on surface
        const matrix_type<D, 1>& meas_local =
            trk_state.template measurement_local<D>();

        // Predicted vector of bound track parameters
        const matrix_type<e_bound_size, 1>& predicted_vec =
            bound_params.vector();

        // Predicted covaraince of bound track parameters
        const matrix_type<e_bound_size, e_bound_size>& predicted_cov =
            bound_params.covariance();

        const matrix_type<e_bound_size, e_bound_size> predicted_cov_inv =
            matrix_operator().inverse(predicted_cov);
        const matrix_type<e_bound_size, e_bound_size> filtered_cov_inv =
            matrix_operator().inverse(trk_state.filtered().covariance());

        // Eq (5.1-11) of "Applied Optimal Estimation"
        const matrix_type<e_bound_size, e_bound_size> smoothed_cov_inv =
            predicted_cov_inv + filtered_cov_inv;

        const matrix_type<e_bound_size, e_bound_size> smoothed_cov =
            matrix_operator().inverse(smoothed_cov_inv);

        // Eq (5.1-12) of "Applied Optimal Estimation"
        const matrix_type<e_bound_size, 1u> smoothed_vec =
            smoothed_cov * (filtered_cov_inv * trk_state.filtered().vector() +
                            predicted_cov_inv * predicted_vec);

        trk_state.smoothed().set_vector(smoothed_vec);
        trk_state.smoothed().set_covariance(smoothed_cov);

        const matrix_type<D, 1> residual_smt = meas_local - H * smoothed_vec;

        // Spatial resolution (Measurement covariance)
        const matrix_type<D, D> V =
            trk_state.template measurement_covariance<D>();

        // Eq (3.39) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<D, D> R_smt =
            V - H * smoothed_cov * matrix_operator().transpose(H);

        // Eq (3.40) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<1, 1> chi2_smt =
            matrix_operator().transpose(residual_smt) *
            matrix_operator().inverse(R_smt) * residual_smt;

        trk_state.smoothed_chi2() = matrix_operator().element(chi2_smt, 0, 0);

        return true;
    }
};

}  // namespace traccc
