/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/status_codes.hpp"

// Detray inlcude(s)
#include <detray/geometry/shapes/line.hpp>

namespace traccc {

/// Type unrolling functor to smooth the track parameters after the Kalman
/// filtering
template <typename algebra_t>
struct gain_matrix_smoother {

    // Type declarations
    using scalar_type = detray::dscalar<algebra_t>;
    using size_type = detray::dsize_type<algebra_t>;
    template <size_type ROWS, size_type COLS>
    using matrix_type = detray::dmatrix<algebra_t, ROWS, COLS>;
    using bound_vector_type = traccc::bound_vector<algebra_t>;
    using bound_matrix_type = traccc::bound_matrix<algebra_t>;

    /// Gain matrix smoother operation
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param mask_group mask group that contains the mask of the current
    /// surface
    /// @param index mask index of the current surface
    /// @param cur_state track state of the current surface
    /// @param next_state track state of the next surface
    ///
    /// @return true if the update succeeds
    template <typename mask_group_t, typename index_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status operator()(
        const mask_group_t& /*mask_group*/, const index_t& /*index*/,
        track_state<algebra_t>& cur_state,
        const track_state<algebra_t>& next_state) {

        using shape_type = typename mask_group_t::value_type::shape;

        const auto D = cur_state.get_measurement().meas_dim;
        assert(D == 1u || D == 2u);
        if (D == 1u) {
            return smoothe<1u, shape_type>(cur_state, next_state);
        } else if (D == 2u) {
            return smoothe<2u, shape_type>(cur_state, next_state);
        }

        return kalman_fitter_status::ERROR_OTHER;
    }

    template <size_type D, typename shape_t>
    [[nodiscard]] TRACCC_HOST_DEVICE inline kalman_fitter_status smoothe(
        track_state<algebra_t>& cur_state,
        const track_state<algebra_t>& next_state) const {
        const auto meas = cur_state.get_measurement();

        static_assert(((D == 1u) || (D == 2u)),
                      "The measurement dimension should be 1 or 2");

        const auto& next_smoothed = next_state.smoothed();
        const auto& next_predicted = next_state.predicted();
        const auto& cur_filtered = cur_state.filtered();

        // Next track state parameters
        const bound_matrix_type& next_jacobian = next_state.jacobian();
        const bound_vector_type& next_smoothed_vec = next_smoothed.vector();
        const bound_matrix_type& next_smoothed_cov = next_smoothed.covariance();
        const bound_vector_type& next_predicted_vec = next_predicted.vector();
        const bound_matrix_type& next_predicted_cov =
            next_predicted.covariance();

        // Current track state parameters
        const bound_vector_type& cur_filtered_vec = cur_filtered.vector();
        const bound_matrix_type& cur_filtered_cov = cur_filtered.covariance();

        // Regularization matrix for numerical stability
        constexpr scalar_type epsilon = 1e-13f;
        const auto regularization =
            matrix::identity<bound_matrix_type>() * epsilon;
        const bound_matrix_type regularized_predicted_cov =
            next_predicted_cov + regularization;

        // Calculate smoothed parameter for current state
        const bound_matrix_type A = cur_filtered_cov *
                                    matrix::transpose(next_jacobian) *
                                    matrix::inverse(regularized_predicted_cov);

        const bound_vector_type smt_vec =
            cur_filtered_vec + A * (next_smoothed_vec - next_predicted_vec);
        const bound_matrix_type smt_cov =
            cur_filtered_cov +
            A * (next_smoothed_cov - next_predicted_cov) * matrix::transpose(A);

        cur_state.smoothed().set_vector(smt_vec);
        cur_state.smoothed().set_covariance(smt_cov);

        // Return false if track is parallel to z-axis or phi is not finite
        const scalar theta = cur_state.smoothed().theta();

        if (theta <= 0.f || theta >= constant<traccc::scalar>::pi) {
            return kalman_fitter_status::ERROR_THETA_ZERO;
        }

        if (!std::isfinite(cur_state.smoothed().phi())) {
            return kalman_fitter_status::ERROR_INVERSION;
        }

        if (std::abs(cur_state.smoothed().qop()) == 0.f) {
            return kalman_fitter_status::ERROR_QOP_ZERO;
        }

        // Wrap the phi in the range of [-pi, pi]
        wrap_phi(cur_state.smoothed());

        matrix_type<D, e_bound_size> H = meas.subs.template projector<D>();

        // Correct sign for line detector
        if constexpr (std::is_same_v<shape_t, detray::line<true>> ||
                      std::is_same_v<shape_t, detray::line<false>>) {

            if (getter::element(smt_vec, e_bound_loc0, 0u) < 0) {
                getter::element(H, 0u, e_bound_loc0) = -1;
            }
        }

        // Calculate smoothed chi square
        const matrix_type<D, 1>& meas_local =
            cur_state.template measurement_local<D>();
        const matrix_type<D, D>& V =
            cur_state.template measurement_covariance<D>();
        const matrix_type<D, 1> residual = meas_local - H * smt_vec;
        const matrix_type<D, D> R = V - H * smt_cov * matrix::transpose(H);
        const matrix_type<1, 1> chi2 =
            matrix::transpose(residual) * matrix::inverse(R) * residual;

        if (getter::element(chi2, 0, 0) < 0.f) {
            return kalman_fitter_status::ERROR_SMOOTHER_CHI2_NEGATIVE;
        }

        cur_state.smoothed_chi2() = getter::element(chi2, 0, 0);
        cur_state.is_smoothed = true;

        return kalman_fitter_status::SUCCESS;
    }
};

}  // namespace traccc
