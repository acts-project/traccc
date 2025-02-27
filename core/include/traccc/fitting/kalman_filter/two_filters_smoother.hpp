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
#include "traccc/fitting/status_codes.hpp"

namespace traccc {

/// Type unrolling functor for two-filters smoother
template <typename algebra_t>
struct two_filters_smoother {

    // Type declarations
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
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status operator()(
        const mask_group_t& /*mask_group*/, const index_t& /*index*/,
        track_state<algebra_t>& trk_state,
        bound_track_parameters<algebra_t>& bound_params) const {

        using shape_type = typename mask_group_t::value_type::shape;

        const auto D = trk_state.get_measurement().meas_dim;
        assert(D == 1u || D == 2u);
        if (D == 1u) {
            return smoothe<1u, shape_type>(trk_state, bound_params);
        } else if (D == 2u) {
            return smoothe<2u, shape_type>(trk_state, bound_params);
        }

        return kalman_fitter_status::ERROR_OTHER;
    }

    // Reference: The Optimun Linear Smoother as a Combination of Two Optimum
    // Linear Filters
    template <size_type D, typename shape_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status smoothe(
        track_state<algebra_t>& trk_state,
        bound_track_parameters<algebra_t>& bound_params) const {

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
        const matrix_type<e_bound_size, 1> predicted_vec =
            bound_params.vector();

        // Predicted covaraince of bound track parameters
        const matrix_type<e_bound_size, e_bound_size> predicted_cov =
            bound_params.covariance();

        const matrix_type<e_bound_size, e_bound_size> predicted_cov_inv =
            matrix::inverse(predicted_cov);
        const matrix_type<e_bound_size, e_bound_size> filtered_cov_inv =
            matrix::inverse(trk_state.filtered().covariance());

        // Eq (3.38) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<e_bound_size, e_bound_size> smoothed_cov_inv =
            predicted_cov_inv + filtered_cov_inv;

        const matrix_type<e_bound_size, e_bound_size> smoothed_cov =
            matrix::inverse(smoothed_cov_inv);

        // Eq (3.38) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
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
            V - H * smoothed_cov * matrix::transpose(H);

        // Eq (3.40) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<1, 1> chi2_smt = matrix::transpose(residual_smt) *
                                           matrix::inverse(R_smt) *
                                           residual_smt;

        if (getter::element(chi2_smt, 0, 0) < 0.f) {
            return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NEGATIVE;
        }

        trk_state.smoothed_chi2() = getter::element(chi2_smt, 0, 0);

        /*************************************
         *  Set backward filtered parameter
         *************************************/

        const auto I66 =
            matrix::identity<matrix_type<e_bound_size, e_bound_size>>();
        const auto I_m = matrix::identity<matrix_type<D, D>>();

        const matrix_type<D, D> M =
            H * predicted_cov * matrix::transpose(H) + V;

        // Kalman gain matrix
        const matrix_type<6, D> K =
            predicted_cov * matrix::transpose(H) * matrix::inverse(M);

        // Calculate the filtered track parameters
        const matrix_type<6, 1> filtered_vec =
            predicted_vec + K * (meas_local - H * predicted_vec);
        const matrix_type<6, 6> filtered_cov = (I66 - K * H) * predicted_cov;

        // Residual between measurement and (projected) filtered vector
        const matrix_type<D, 1> residual = meas_local - H * filtered_vec;

        // Calculate backward chi2
        const matrix_type<D, D> R = (I_m - H * K) * V;
        const matrix_type<1, 1> chi2 =
            matrix::transpose(residual) * matrix::inverse(R) * residual;

        // Update the bound track parameters
        bound_params.set_vector(filtered_vec);
        bound_params.set_covariance(filtered_cov);

        // Return false if track is parallel to z-axis or phi is not finite
        const scalar theta = bound_params.theta();
        if (theta <= 0.f || theta >= constant<traccc::scalar>::pi) {
            return kalman_fitter_status::ERROR_THETA_ZERO;
        }

        if (!std::isfinite(bound_params.phi())) {
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (std::abs(bound_params.qop()) == 0.f) {
            return kalman_fitter_status::ERROR_QOP_ZERO;
        }

        if (getter::element(chi2, 0, 0) < 0.f) {
            return kalman_fitter_status::ERROR_UPDATER_CHI2_NEGATIVE;
        }

        // Set backward chi2
        trk_state.backward_chi2() = getter::element(chi2, 0, 0);

        // Wrap the phi in the range of [-pi, pi]
        wrap_phi(bound_params);

        trk_state.is_smoothed = true;
        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
