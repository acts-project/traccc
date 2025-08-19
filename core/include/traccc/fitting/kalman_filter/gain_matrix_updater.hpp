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
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/measurement_helpers.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/fitting/status_codes.hpp"

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
        const measurement_collection_types::const_device& measurements,
        const bound_track_parameters<algebra_t>& bound_params,
        const bool is_line) const {

        const auto D = measurements.at(trk_state.measurement_index()).meas_dim;

        assert(D == 1u || D == 2u);

        return update(trk_state, measurements, bound_params, D, is_line);
    }

    template <typename track_state_backend_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status update(
        typename edm::track_state<track_state_backend_t>& trk_state,
        const measurement_collection_types::const_device& measurements,
        const bound_track_parameters<algebra_t>& bound_params,
        const unsigned int dim, const bool is_line) const {

        static constexpr unsigned int D = 2;

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

        // Predicted vector of bound track parameters
        const bound_vector_type& predicted_vec = bound_params.vector();

        // Predicted covaraince of bound track parameters
        const bound_matrix_type& predicted_cov = bound_params.covariance();

        matrix_type<D, e_bound_size> H =
            measurements.at(trk_state.measurement_index())
                .subs.template projector<D>();

        // Flip the sign of projector matrix element in case the first element
        // of line measurement is negative
        if (is_line && getter::element(predicted_vec, e_bound_loc0, 0u) < 0) {
            getter::element(H, 0u, e_bound_loc0) = -1;
        }

        if (dim == 1) {
            getter::element(H, 1u, 0u) = 0.f;
            getter::element(H, 1u, 1u) = 0.f;
        }

        // Spatial resolution (Measurement covariance)
        matrix_type<D, D> V;
        edm::get_measurement_covariance<algebra_t>(
            measurements.at(trk_state.measurement_index()), V);

        if (dim == 1) {
            getter::element(V, 1u, 1u) = 1.f;
        }

        const matrix_type<e_bound_size, D> projected_cov =
            algebra::matrix::transposed_product<false, true>(predicted_cov, H);

        const matrix_type<D, D> M = H * projected_cov + V;

        // Kalman gain matrix
        assert(matrix::determinant(M) != 0.f);
        const matrix_type<6, D> K = projected_cov * matrix::inverse(M);

        // Calculate the filtered track parameters
        const matrix_type<6, 1> filtered_vec =
            predicted_vec + K * (meas_local - H * predicted_vec);
        const matrix_type<6, 6> filtered_cov = (I66 - K * H) * predicted_cov;

        // Residual between measurement and (projected) filtered vector
        const matrix_type<D, 1> residual = meas_local - H * filtered_vec;

        // Calculate the chi square
        const matrix_type<D, D> R = (I_m - H * K) * V;
        const matrix_type<1, 1> chi2 =
            algebra::matrix::transposed_product<true, false>(
                residual, matrix::inverse(R)) *
            residual;

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

        // Set the track state parameters
        trk_state.filtered_params().set_vector(filtered_vec);
        trk_state.filtered_params().set_covariance(filtered_cov);
        trk_state.filtered_chi2() = getter::element(chi2, 0, 0);

        // Wrap the phi in the range of [-pi, pi]
        wrap_phi(trk_state.filtered_params());

        assert(!trk_state.filtered_params().is_invalid());

        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
