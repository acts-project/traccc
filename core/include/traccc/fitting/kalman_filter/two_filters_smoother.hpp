/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/measurement_helpers.hpp"
#include "traccc/edm/track_state_collection.hpp"
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
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status operator()(
        typename edm::track_state_collection<algebra_t>::device::proxy_type&
            trk_state,
        const measurement_collection_types::const_device& measurements,
        bound_track_parameters<algebra_t>& bound_params,
        const bool is_line) const {

        const auto D = measurements.at(trk_state.measurement_index()).meas_dim;
        assert(D == 1u || D == 2u);

        return smoothe(trk_state, measurements, bound_params, D, is_line);
    }

    // Reference: The Optimun Linear Smoother as a Combination of Two Optimum
    // Linear Filters
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status smoothe(
        typename edm::track_state_collection<algebra_t>::device::proxy_type&
            trk_state,
        const measurement_collection_types::const_device& measurements,
        bound_track_parameters<algebra_t>& bound_params, const unsigned int dim,
        const bool is_line) const {

        static constexpr unsigned int D = 2;

        assert(dim == 1u || dim == 2u);

        assert(!bound_params.is_invalid());
        assert(!bound_params.surface_link().is_invalid());
        assert(trk_state.filtered_params().surface_link() ==
               bound_params.surface_link());

        // Do not smoothe if the forward pass produced an error
        if (trk_state.filtered_params().is_invalid()) {
            return kalman_fitter_status::ERROR_INVALID_TRACK_STATE;
        }

        // Measurement data on surface
        matrix_type<D, 1> meas_local;
        edm::get_measurement_local<algebra_t>(
            measurements.at(trk_state.measurement_index()), meas_local);

        assert((dim > 1) || (getter::element(meas_local, 1u, 0u) == 0.f));

        // Predicted vector of bound track parameters
        const matrix_type<e_bound_size, 1>& predicted_vec =
            bound_params.vector();

        // Predicted covaraince of bound track parameters
        const matrix_type<e_bound_size, e_bound_size>& predicted_cov =
            bound_params.covariance();

        const matrix_type<e_bound_size, e_bound_size> predicted_cov_inv =
            matrix::inverse(predicted_cov);
        const matrix_type<e_bound_size, e_bound_size> filtered_cov_inv =
            matrix::inverse(trk_state.filtered_params().covariance());

        // Eq (3.38) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<e_bound_size, e_bound_size> smoothed_cov_inv =
            predicted_cov_inv + filtered_cov_inv;

        assert(matrix::determinant(smoothed_cov_inv) != 0.f);
        const matrix_type<e_bound_size, e_bound_size> smoothed_cov =
            matrix::inverse(smoothed_cov_inv);

        // Eq (3.38) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<e_bound_size, 1u> smoothed_vec =
            smoothed_cov *
            (filtered_cov_inv * trk_state.filtered_params().vector() +
             predicted_cov_inv * predicted_vec);

        trk_state.smoothed_params().set_vector(smoothed_vec);
        trk_state.smoothed_params().set_covariance(smoothed_cov);

        matrix_type<D, e_bound_size> H =
            measurements.at(trk_state.measurement_index())
                .subs.template projector<D>();
        if (dim == 1) {
            getter::element(H, 1u, 0u) = 0.f;
            getter::element(H, 1u, 1u) = 0.f;
        }

        const matrix_type<D, 1> residual_smt = meas_local - H * smoothed_vec;

        // Spatial resolution (Measurement covariance)
        matrix_type<D, D> V;
        edm::get_measurement_covariance<algebra_t>(
            measurements.at(trk_state.measurement_index()), V);
        if (dim == 1) {
            getter::element(V, 1u, 1u) = 1.f;
        }

        // Eq (3.39) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<D, D> R_smt =
            V - H * algebra::matrix::transposed_product<false, true>(
                        smoothed_cov, H);

        // Eq (3.40) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        assert(matrix::determinant(R_smt) != 0.f);
        const matrix_type<1, 1> chi2_smt =
            algebra::matrix::transposed_product<true, false>(
                residual_smt, matrix::inverse(R_smt)) *
            residual_smt;

        if (getter::element(chi2_smt, 0, 0) < 0.f) {
            return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NEGATIVE;
        }

        if (!std::isfinite(getter::element(chi2_smt, 0, 0))) {
            return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NOT_FINITE;
        }

        trk_state.smoothed_chi2() = getter::element(chi2_smt, 0, 0);

        /*************************************
         *  Set backward filtered parameter
         *************************************/

        // Flip the sign of projector matrix element in case the first element
        // of line measurement is negative
        if (is_line && getter::element(predicted_vec, e_bound_loc0, 0u) < 0) {
            getter::element(H, 0u, e_bound_loc0) = -1;
        }

        const auto I66 =
            matrix::identity<matrix_type<e_bound_size, e_bound_size>>();
        const auto I_m = matrix::identity<matrix_type<D, D>>();

        const matrix_type<e_bound_size, D> projected_cov =
            algebra::matrix::transposed_product<false, true>(predicted_cov, H);

        const matrix_type<D, D> M = H * projected_cov + V;

        // Kalman gain matrix
        assert(matrix::determinant(M) != 0.f);
        assert(std::isfinite(matrix::determinant(M)));
        const matrix_type<6, D> K = projected_cov * matrix::inverse(M);

        // Calculate the filtered track parameters
        const matrix_type<6, 1> filtered_vec =
            predicted_vec + K * (meas_local - H * predicted_vec);
        const matrix_type<6, 6> filtered_cov = (I66 - K * H) * predicted_cov;

        // Residual between measurement and (projected) filtered vector
        const matrix_type<D, 1> residual = meas_local - H * filtered_vec;

        // Calculate backward chi2
        const matrix_type<D, D> R = (I_m - H * K) * V;
        // assert(matrix::determinant(R) != 0.f);
        assert(std::isfinite(matrix::determinant(R)));
        const matrix_type<1, 1> chi2 =
            algebra::matrix::transposed_product<true, false>(
                residual, matrix::inverse(R)) *
            residual;

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

        if (!std::isfinite(getter::element(chi2, 0, 0))) {
            return kalman_fitter_status::ERROR_UPDATER_CHI2_NOT_FINITE;
        }

        // Set backward chi2
        trk_state.backward_chi2() = getter::element(chi2, 0, 0);

        // Wrap the phi in the range of [-pi, pi]
        wrap_phi(bound_params);

        trk_state.set_smoothed();

        assert(!bound_params.is_invalid());

        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
