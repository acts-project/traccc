/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"

namespace traccc {

/// Type unrolling functor for Kalman updating
template <typename algebra_t>
struct gain_matrix_updater {

    // Type declarations
    using output_type = bool;
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
    /// @param propagation propagator state
    ///
    /// @return true if the update succeeds
    template <typename mask_group_t, typename index_t,
              typename propagator_state_t>
    TRACCC_HOST_DEVICE inline output_type operator()(
        const mask_group_t& mask_group, const index_t& index,
        track_state<algebra_t>& trk_state,
        propagator_state_t& propagation) const {

        auto& stepping = propagation._stepping;

        // Some identity matrices
        // @Note: Make constexpr work
        static const matrix_type<6, 6> I66 =
            matrix_operator().template identity<e_bound_size, e_bound_size>();
        static const matrix_type<2, 2> I22 =
            matrix_operator().template identity<2, 2>();

        // projection matrix
        const matrix_type<2, 6> H =
            mask_group[index].template projection_matrix<e_bound_size>();

        // Measurement data on surface
        const matrix_type<2, 1>& meas_local = trk_state.measurement_local();

        // Predicted vector of bound track parameters
        const matrix_type<6, 1>& predicted_vec =
            stepping._bound_params.vector();

        // Predicted covaraince of bound track parameters
        const matrix_type<6, 6>& predicted_cov =
            stepping._bound_params.covariance();

        // Set track state parameters
        trk_state.predicted().set_vector(predicted_vec);
        trk_state.predicted().set_covariance(predicted_cov);

        // Spatial resolution (Measurement covariance)
        const matrix_type<2, 2> V = trk_state.measurement_covariance();

        const matrix_type<2, 2> M =
            H * predicted_cov * matrix_operator().transpose(H) + V;

        // Kalman gain matrix
        const matrix_type<6, 2> K = predicted_cov *
                                    matrix_operator().transpose(H) *
                                    matrix_operator().inverse(M);

        // Calculate the filtered track parameters
        const matrix_type<6, 1> filtered_vec =
            predicted_vec + K * (meas_local - H * predicted_vec);
        const matrix_type<6, 6> filtered_cov = (I66 - K * H) * predicted_cov;

        // Residual between measurement and (projected) filtered vector
        const matrix_type<2, 1> residual = meas_local - H * filtered_vec;

        // Calculate the chi square
        const matrix_type<2, 2> R = (I22 - H * K) * V;
        const matrix_type<1, 1> chi2 = matrix_operator().transpose(residual) *
                                       matrix_operator().inverse(R) * residual;

        // Set the stepper parameter
        stepping._bound_params.set_vector(filtered_vec);
        stepping._bound_params.set_covariance(filtered_cov);

        // Set the track state parameters
        trk_state.filtered().set_vector(filtered_vec);
        trk_state.filtered().set_covariance(filtered_cov);
        trk_state.filtered_chi2() = matrix_operator().element(chi2, 0, 0);

        return true;
    }
};

}  // namespace traccc
