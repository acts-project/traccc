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
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/measurement_helpers.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/fitting/details/regularize_covariance.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/logging.hpp"

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
        const typename edm::measurement_collection<algebra_t>::const_device&
            measurements,
        bound_track_parameters<algebra_t>& bound_params,
        const bool is_line) const {

        static constexpr unsigned int D = 2;

        [[maybe_unused]] const unsigned int dim{
            measurements.at(trk_state.measurement_index()).dimensions()};

        assert(dim == 1u || dim == 2u);

        assert(!bound_params.is_invalid());
        assert(!bound_params.surface_link().is_invalid());
        assert(trk_state.filtered_params().surface_link() ==
               bound_params.surface_link());

        // Do not smoothe if the forward pass produced an error
        if (trk_state.filtered_params().is_invalid()) {
            TRACCC_ERROR_HOST_DEVICE("Filtered track state invalid");
            TRACCC_ERROR_HOST(trk_state.filtered_params());
            return kalman_fitter_status::ERROR_UPDATER_SKIPPED_STATE;
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
        matrix_type<e_bound_size, e_bound_size> smoothed_cov =
            matrix::inverse(smoothed_cov_inv);

        // Check the covariance for consistency
        // @TODO: Need to understand why negative variance happens
        if (constexpr traccc::scalar min_var{-0.01f};
            !details::regularize_covariance<algebra_t>(smoothed_cov, min_var)) {
            TRACCC_ERROR_HOST_DEVICE("Negative variance after smoothing");
            return kalman_fitter_status::ERROR_SMOOTHER_INVALID_COVARIANCE;
        }

        // Eq (3.38) of "Pattern Recognition, Tracking and Vertex
        // Reconstruction in Particle Detectors"
        const matrix_type<e_bound_size, 1u> smoothed_vec =
            smoothed_cov *
            (filtered_cov_inv * trk_state.filtered_params().vector() +
             predicted_cov_inv * predicted_vec);

        trk_state.smoothed_params().set_vector(smoothed_vec);

        // Return false if track is parallel to z-axis or phi is not finite
        if (!std::isfinite(trk_state.smoothed_params().theta())) {
            TRACCC_ERROR_HOST_DEVICE(
                "Theta is infinite after smoothing (Matrix inversion)");
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (!std::isfinite(trk_state.smoothed_params().phi())) {
            TRACCC_ERROR_HOST_DEVICE(
                "Phi is infinite after smoothing (Matrix inversion)");
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (math::fabs(trk_state.smoothed_params().qop()) == 0.f) {
            TRACCC_ERROR_HOST_DEVICE("q/p is zero after smoothing");
            return kalman_fitter_status::ERROR_QOP_ZERO;
        }

        trk_state.smoothed_params().set_covariance(smoothed_cov);

        // Wrap the phi and theta angles in their valid ranges
        normalize_angles(trk_state.smoothed_params());

        const subspace<algebra_t, e_bound_size> subs(
            measurements.at(trk_state.measurement_index()).subspace());
        matrix_type<D, e_bound_size> H = subs.template projector<D>();
        // @TODO: Fix properly
        if (getter::element(meas_local, 1u, 0u) == 0.f /*dim == 1*/) {
            getter::element(H, 1u, 0u) = 0.f;
            getter::element(H, 1u, 1u) = 0.f;
        }

        const matrix_type<D, 1> residual_smt = meas_local - H * smoothed_vec;

        // Spatial resolution (Measurement covariance)
        matrix_type<D, D> V;
        edm::get_measurement_covariance<algebra_t>(
            measurements.at(trk_state.measurement_index()), V);
        // @TODO: Fix properly
        if (getter::element(meas_local, 1u, 0u) == 0.f /*dim == 1*/) {
            getter::element(V, 1u, 1u) = 1000.f;
        }

        TRACCC_DEBUG_HOST("Measurement position: " << meas_local);
        TRACCC_DEBUG_HOST("Measurement variance:\n" << V);
        TRACCC_DEBUG_HOST("Predicted residual: " << meas_local -
                                                        H * predicted_vec);

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

        const scalar chi2_smt_value{getter::element(chi2_smt, 0, 0)};

        TRACCC_VERBOSE_HOST("Smoothed residual: " << residual_smt);
        TRACCC_DEBUG_HOST("R_smt:\n" << R_smt);
        TRACCC_DEBUG_HOST_DEVICE("det(R_smt): %f", matrix::determinant(R_smt));
        TRACCC_DEBUG_HOST("R_smt_inv:\n" << matrix::inverse(R_smt));
        TRACCC_VERBOSE_HOST_DEVICE("Smoothed chi2: %f", chi2_smt_value);

        if (chi2_smt_value < 0.f) {
            TRACCC_ERROR_HOST_DEVICE("Smoothed chi2 negative: %f",
                                     chi2_smt_value);

            // @TODO: Need to understand why negative chi2 happens
            if (chi2_smt_value < -10.f) {
                return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NEGATIVE;
            }
        }

        if (!std::isfinite(chi2_smt_value)) {
            TRACCC_ERROR_HOST_DEVICE("Smoothed chi2 infinite");
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

        TRACCC_DEBUG_HOST("H:\n" << H);
        TRACCC_DEBUG_HOST("K:\n" << K);

        // Calculate the filtered track parameters
        const matrix_type<6, 1> filtered_vec =
            predicted_vec + K * (meas_local - H * predicted_vec);
        const matrix_type<6, 6> i_minus_kh = I66 - K * H;
        matrix_type<6, 6> filtered_cov =
            i_minus_kh * predicted_cov * matrix::transpose(i_minus_kh) +
            K * V * matrix::transpose(K);

        // Check the covariance for consistency
        // @TODO: Need to understand why negative variance happens
        if (constexpr traccc::scalar min_var{-0.01f};
            !details::regularize_covariance<algebra_t>(filtered_cov, min_var)) {
            TRACCC_ERROR_HOST_DEVICE("Negative variance after filtering");
            return kalman_fitter_status::ERROR_SMOOTHER_INVALID_COVARIANCE;
        }

        // Update the bound track parameters
        bound_params.set_vector(filtered_vec);

        // Return false if track is parallel to z-axis or phi is not finite
        if (!std::isfinite(bound_params.theta())) {
            TRACCC_ERROR_HOST_DEVICE(
                "Theta is infinite after filering in smoother (Matrix "
                "inversion)");
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (!std::isfinite(bound_params.phi())) {
            TRACCC_ERROR_HOST_DEVICE(
                "Phi is infinite after filering in smoother (Matrix "
                "inversion)");
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (math::fabs(bound_params.qop()) == 0.f) {
            TRACCC_ERROR_HOST_DEVICE("q/p is zero after filering in smoother");
            return kalman_fitter_status::ERROR_QOP_ZERO;
        }

        bound_params.set_covariance(filtered_cov);

        // Residual between measurement and (projected) filtered vector
        const matrix_type<D, 1> residual = meas_local - H * filtered_vec;

        // Calculate backward chi2
        const matrix_type<D, D> R = (I_m - H * K) * V;
        // assert(matrix::determinant(R) != 0.f); // @TODO: This fails
        assert(std::isfinite(matrix::determinant(R)));
        const matrix_type<1, 1> chi2 =
            algebra::matrix::transposed_product<true, false>(
                residual, matrix::inverse(R)) *
            residual;

        const scalar chi2_val{getter::element(chi2, 0, 0)};

        TRACCC_VERBOSE_HOST("Filtered residual: " << residual);
        TRACCC_DEBUG_HOST("R:\n" << R);
        TRACCC_DEBUG_HOST_DEVICE("det(R): %f", matrix::determinant(R));
        TRACCC_DEBUG_HOST("R_inv:\n" << matrix::inverse(R));
        TRACCC_VERBOSE_HOST_DEVICE("Filtered chi2: %f", chi2_val);

        if (chi2_val < 0.f) {
            TRACCC_ERROR_HOST_DEVICE("Filtered chi2 negative: %f", chi2_val);
            return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NEGATIVE;
        }

        if (!std::isfinite(chi2_val)) {
            TRACCC_ERROR_HOST_DEVICE("Filtered chi2 infinite");
            return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NOT_FINITE;
        }

        // Set backward chi2
        trk_state.backward_chi2() = chi2_val;

        // Wrap the phi and theta angles in their valid ranges
        normalize_angles(bound_params);

        const scalar theta = bound_params.theta();
        if (theta <= 0.f || theta >= 2.f * constant<traccc::scalar>::pi) {
            TRACCC_ERROR_HOST_DEVICE("Hit theta pole after smoothing : %f",
                                     theta);
            return kalman_fitter_status::ERROR_THETA_POLE;
        }

        trk_state.set_smoothed();

        assert(!bound_params.is_invalid());

        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
