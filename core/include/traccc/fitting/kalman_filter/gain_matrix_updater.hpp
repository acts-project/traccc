/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
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
#include "traccc/utils/subspace.hpp"

namespace traccc {

/// Type unrolling functor for Kalman updating
template <typename algebra_t>
struct gain_matrix_updater {

    // Type declarations
    using size_type = detray::dsize_type<algebra_t>;
    template <size_type ROWS, size_type COLS>
    using matrix_type = detray::dmatrix<algebra_t, ROWS, COLS>;
    using bound_vector_type = traccc::bound_vector<algebra_t>;
    using bound_matrix_type = traccc::bound_matrix<algebra_t>;

    /// Gain matrix updater operation
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param mask_group mask group that contains the mask of surface
    /// @param index mask index of surface
    /// @param trk_state track state of the surface
    /// @param bound_params bound parameter
    ///
    /// @return true if the update succeeds
    template <typename track_state_backend_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status operator()(
        typename edm::track_state<track_state_backend_t>& trk_state,
        const edm::measurement_collection<default_algebra>::const_device&
            measurements,
        const bound_track_parameters<algebra_t>& bound_params,
        const bool is_line) const {

        static constexpr unsigned int D = 2;

        [[maybe_unused]] const unsigned int dim{
            measurements.at(trk_state.measurement_index()).dimensions()};

        TRACCC_VERBOSE_HOST_DEVICE("In gain-matrix-updater...");
        TRACCC_VERBOSE_HOST_DEVICE("Measurement dim: %d", dim);

        assert(dim == 1u || dim == 2u);

        assert(!bound_params.is_invalid());
        assert(!bound_params.surface_link().is_invalid());

        // Some identity matrices
        // @TODO: Make constexpr work
        const auto I66 = matrix::identity<bound_matrix_type>();
        const auto I_m = matrix::identity<matrix_type<D, D>>();

        // Measurement data on surface
        matrix_type<D, 1> meas_local;
        edm::get_measurement_local<algebra_t>(
            measurements.at(trk_state.measurement_index()), meas_local);

        assert((dim > 1) || (getter::element(meas_local, 1u, 0u) == 0.f));

        TRACCC_DEBUG_HOST("Predicted param.: " << bound_params);

        // Predicted vector of bound track parameters
        const bound_vector_type& predicted_vec = bound_params.vector();

        // Predicted covaraince of bound track parameters
        const bound_matrix_type& predicted_cov = bound_params.covariance();

        const subspace<algebra_t, e_bound_size> subs(
            measurements.at(trk_state.measurement_index()).subspace());
        matrix_type<D, e_bound_size> H = subs.template projector<D>();

        // Flip the sign of projector matrix element in case the first element
        // of line measurement is negative
        if (is_line && getter::element(predicted_vec, e_bound_loc0, 0u) < 0) {
            getter::element(H, 0u, e_bound_loc0) = -1;
        }

        // @TODO: Fix properly
        if (/*dim == 1*/ getter::element(meas_local, 1u, 0u) == 0.f) {
            getter::element(H, 1u, 0u) = 0.f;
            getter::element(H, 1u, 1u) = 0.f;
        }

        // Spatial resolution (Measurement covariance)
        matrix_type<D, D> V;
        edm::get_measurement_covariance<algebra_t>(
            measurements.at(trk_state.measurement_index()), V);
        // @TODO: Fix properly
        if (/*dim == 1*/ getter::element(meas_local, 1u, 0u) == 0.f) {
            getter::element(V, 1u, 1u) = 1000.f;
        }

        TRACCC_DEBUG_HOST("Measurement position: " << meas_local);
        TRACCC_DEBUG_HOST("Measurement variance:\n" << V);
        TRACCC_DEBUG_HOST("Predicted residual: " << meas_local -
                                                        H * predicted_vec);

        const matrix_type<e_bound_size, D> projected_cov =
            algebra::matrix::transposed_product<false, true>(predicted_cov, H);

        const matrix_type<D, D> M = H * projected_cov + V;

        // Kalman gain matrix
        assert(matrix::determinant(M) != 0.f);
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

        TRACCC_DEBUG_HOST("Filtered param:\n" << filtered_vec);
        TRACCC_DEBUG_HOST("Filtered cov:\n" << filtered_cov);

        // Check the covariance for consistency
        // @TODO: Need to understand why negative variance happens
        if (constexpr traccc::scalar min_var{-0.01f};
            !details::regularize_covariance<algebra_t>(filtered_cov, min_var)) {
            TRACCC_ERROR_HOST_DEVICE("Negative variance after filtering");
            return kalman_fitter_status::ERROR_UPDATER_INVALID_COVARIANCE;
        }

        // Return false if track is parallel to z-axis or phi is not finite
        if (!std::isfinite(getter::element(filtered_vec, e_bound_theta, 0))) {
            TRACCC_ERROR_HOST_DEVICE(
                "Theta is infinite after filtering (Matrix inversion)");
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (!std::isfinite(getter::element(filtered_vec, e_bound_phi, 0))) {
            TRACCC_ERROR_HOST_DEVICE(
                "Phi is infinite after filtering (Matrix inversion)");
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (math::fabs(getter::element(filtered_vec, e_bound_qoverp, 0)) ==
            0.f) {
            TRACCC_ERROR_HOST_DEVICE("q/p is zero after filtering");
            return kalman_fitter_status::ERROR_QOP_ZERO;
        }

        // Residual between measurement and (projected) filtered vector
        const matrix_type<D, 1> residual = meas_local - H * filtered_vec;

        // Calculate the chi square
        const matrix_type<D, D> R = (I_m - H * K) * V;
        const matrix_type<1, 1> chi2 =
            algebra::matrix::transposed_product<true, false>(
                residual, matrix::inverse(R)) *
            residual;

        const scalar chi2_val{getter::element(chi2, 0, 0)};

        TRACCC_VERBOSE_HOST("Filtered residual: " << residual);
        TRACCC_DEBUG_HOST("R:\n" << R);
        TRACCC_DEBUG_HOST_DEVICE("det(R): %f", matrix::determinant(R));
        TRACCC_DEBUG_HOST("R_inv:\n" << matrix::inverse(R));
        TRACCC_VERBOSE_HOST_DEVICE("Chi2: %f", chi2_val);

        if (chi2_val < 0.f) {
            TRACCC_ERROR_HOST_DEVICE("Chi2 negative");
            return kalman_fitter_status::ERROR_UPDATER_CHI2_NEGATIVE;
        }

        if (!std::isfinite(chi2_val)) {
            TRACCC_ERROR_HOST_DEVICE("Chi2 infinite");
            return kalman_fitter_status::ERROR_UPDATER_CHI2_NOT_FINITE;
        }

        // Set the chi2 for this track and measurement
        trk_state.filtered_params().set_vector(filtered_vec);
        trk_state.filtered_params().set_covariance(filtered_cov);
        trk_state.filtered_chi2() = chi2_val;

        if (math::fmod(trk_state.filtered_params().theta(),
                       2.f * constant<traccc::scalar>::pi) == 0.f) {
            TRACCC_ERROR_HOST_DEVICE(
                "Hit theta pole after filtering : %f (unrecoverable error "
                "pre-normalization)",
                trk_state.filtered_params().theta());
            return kalman_fitter_status::ERROR_THETA_POLE;
        }

        // Wrap the phi and theta angles in their valid ranges
        normalize_angles(trk_state.filtered_params());

        const scalar theta = trk_state.filtered_params().theta();
        if (theta <= 0.f || theta >= 2.f * constant<traccc::scalar>::pi) {
            TRACCC_ERROR_HOST_DEVICE("Hit theta pole in filtering : %f", theta);
            return kalman_fitter_status::ERROR_THETA_POLE;
        }

        assert(!trk_state.filtered_params().is_invalid());

        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
