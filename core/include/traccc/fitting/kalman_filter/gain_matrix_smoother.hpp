/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"

// detray include(s).
#include "detray/propagator/navigator.hpp"

namespace traccc {

/// Type unrolling functor to smooth the track parameters after the Kalman
/// filtering
template <typename algebra_t>
struct gain_matrix_smoother {

    // Type declarations
    using output_type = bool;
    using matrix_operator = typename algebra_t::matrix_actor;
    using size_type = typename matrix_operator::size_ty;
    template <size_type ROWS, size_type COLS>
    using matrix_type =
        typename matrix_operator::template matrix_type<ROWS, COLS>;

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
    TRACCC_HOST_DEVICE inline output_type operator()(
        const mask_group_t& mask_group, const index_t& index,
        track_state<algebra_t>& cur_state,
        const track_state<algebra_t>& next_state) {

        const auto& next_smoothed = next_state.smoothed();
        const auto& next_predicted = next_state.predicted();
        const auto& cur_filtered = cur_state.filtered();

        // Next track state parameters
        const matrix_type<6, 6>& next_jacobian = next_state.jacobian();
        const matrix_type<6, 1>& next_smoothed_vec = next_smoothed.vector();
        const matrix_type<6, 6>& next_smoothed_cov = next_smoothed.covariance();
        const matrix_type<6, 1>& next_predicted_vec = next_predicted.vector();
        const matrix_type<6, 6>& next_predicted_cov =
            next_predicted.covariance();

        // Current track state parameters
        const matrix_type<6, 1>& cur_filtered_vec = cur_filtered.vector();
        const matrix_type<6, 6>& cur_filtered_cov = cur_filtered.covariance();

        // Regularization matrix for numerical stability
        static constexpr scalar epsilon = 1e-13;
        const matrix_type<6, 6> regularization =
            matrix_operator().template identity<e_bound_size, e_bound_size>() *
            epsilon;
        const matrix_type<6, 6> regularized_predicted_cov =
            next_predicted_cov + regularization;

        // Calculate smoothed parameter for current state
        const matrix_type<6, 6> A =
            cur_filtered_cov * matrix_operator().transpose(next_jacobian) *
            matrix_operator().inverse(regularized_predicted_cov);

        const matrix_type<6, 1> smt_vec =
            cur_filtered_vec + A * (next_smoothed_vec - next_predicted_vec);
        const matrix_type<6, 6> smt_cov =
            cur_filtered_cov + A * (next_smoothed_cov - next_predicted_cov) *
                                   matrix_operator().transpose(A);

        cur_state.smoothed().set_vector(smt_vec);
        cur_state.smoothed().set_covariance(smt_cov);

        // projection matrix
        const matrix_type<2, 6> H =
            mask_group[index].template projection_matrix<e_bound_size>();

        // Calculate smoothed chi square
        const matrix_type<2, 1>& meas_local = cur_state.measurement_local();
        const matrix_type<2, 2>& V = cur_state.measurement_covariance();
        const matrix_type<2, 1> residual = meas_local - H * smt_vec;
        const matrix_type<2, 2> R =
            V - H * smt_cov * matrix_operator().transpose(H);
        const matrix_type<1, 1> chi2 = matrix_operator().transpose(residual) *
                                       matrix_operator().inverse(R) * residual;

        cur_state.smoothed_chi2() = matrix_operator().element(chi2, 0, 0);

        return true;
    }
};

}  // namespace traccc