/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc {

/// Type unrolling functor for Kalman updating
template <typename algebra_t>
struct gain_matrix_updater {

    // Type declarations
    using matrix_operator = typename algebra_t::matrix_actor;
    using size_type = typename matrix_operator::size_ty;
    template <size_type ROWS, size_type COLS>
    using matrix_type =
        typename matrix_operator::template matrix_type<ROWS, COLS>;

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
    template <typename mask_group_t, typename index_t>
    TRACCC_HOST_DEVICE inline void operator()(
        const mask_group_t& mask_group, const index_t& index,
        track_state<algebra_t>& trk_state,
        bound_track_parameters& bound_params) const {

        // Some identity matrices
        // @Note: Make constexpr work
        const matrix_type<6, 6> I66 =
            matrix_operator().template identity<e_bound_size, e_bound_size>();

        constexpr const unsigned int D =
            mask_group_t::value_type::shape::meas_dim;

        const matrix_type<D, D> I_m =
            matrix_operator().template identity<D, D>();

        // projection matrix
        const typename mask_group_t::value_type::projection_matrix_type H =
            mask_group[index].projection_matrix(bound_params);

        // Measurement data on surface
        const matrix_type<D, 1>& meas_local = trk_state.measurement_local();

        // Predicted vector of bound track parameters
        const matrix_type<6, 1>& predicted_vec = bound_params.vector();

        // Predicted covaraince of bound track parameters
        const matrix_type<6, 6>& predicted_cov = bound_params.covariance();

        // Set track state parameters
        trk_state.predicted().set_vector(predicted_vec);
        trk_state.predicted().set_covariance(predicted_cov);

        // Spatial resolution (Measurement covariance)
        const matrix_type<D, D> V = trk_state.measurement_covariance();

        const matrix_type<D, D> M =
            H * predicted_cov * matrix_operator().transpose(H) + V;

        // Kalman gain matrix
        const matrix_type<6, D> K = predicted_cov *
                                    matrix_operator().transpose(H) *
                                    matrix_operator().inverse(M);

        // Calculate the filtered track parameters
        const matrix_type<6, 1> filtered_vec =
            predicted_vec + K * (meas_local - H * predicted_vec);
        const matrix_type<6, 6> filtered_cov = (I66 - K * H) * predicted_cov;

        // Residual between measurement and (projected) filtered vector
        const matrix_type<D, 1> residual = meas_local - H * filtered_vec;

        // Calculate the chi square
        const matrix_type<D, D> R = (I_m - H * K) * V;
        const matrix_type<1, 1> chi2 = matrix_operator().transpose(residual) *
                                       matrix_operator().inverse(R) * residual;

        // Set the stepper parameter
        bound_params.set_vector(filtered_vec);
        bound_params.set_covariance(filtered_cov);

        // Set the track state parameters
        trk_state.filtered().set_vector(filtered_vec);
        trk_state.filtered().set_covariance(filtered_cov);
        trk_state.filtered_chi2() = matrix_operator().element(chi2, 0, 0);
    }
};

}  // namespace traccc
