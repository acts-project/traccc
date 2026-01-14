/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/measurement_helpers.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/subspace.hpp"

namespace traccc {

/// Associate a measurement to a candidate track
struct measurement_selector {

    template <detray::concepts::algebra A, unsigned int ROWS, unsigned int COLS>
    using matrix_t = detray::dmatrix<A, ROWS, COLS>;

    // Where to get the calibration from
    struct config { /*TODO: implement calibration handling*/
    };

    /// Get the obersavtion model for a given measurement
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param measurement the measurement
    /// @param is_line whether the measurement belong to a line surface
    ///
    /// @returns the projection matrix H
    template <detray::concepts::algebra algebra_t, unsigned int D,
              typename measurement_backend_t>
    TRACCC_HOST_DEVICE static constexpr detray::dmatrix<algebra_t, D,
                                                        e_bound_size>
    observation_model(
        const edm::measurement<measurement_backend_t>& measurement,
        const bound_track_parameters<algebra_t>& bound_params,
        const bool is_line) {

        // Oservation model: Subspace of measurment space for this measurement
        const subspace<algebra_t, e_bound_size> subs(measurement.subspace());
        detray::dmatrix<algebra_t, D, e_bound_size> H =
            subs.template projector<D>();

        // Flip the sign of projector matrix element in case the first element
        // of line measurement is negative
        if (is_line && bound_params.bound_local()[e_bound_loc0] < 0) {
            getter::element(H, 0u, e_bound_loc0) = -1;
        }

        if (measurement.dimensions() == 1) {
            getter::element(H, 1u, 0u) = 0.f;
            getter::element(H, 1u, 1u) = 0.f;
        }

        TRACCC_DEBUG_HOST("Observation model (H):\n" << H);

        return H;
    }

    /// Get the calibrated measurement position
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param measurement the measurement
    /// @param is_line whether the measurement belong to a line surface
    /// @param cfg how to apply calibrations
    ///
    /// @returns the projection matrix H
    template <detray::concepts::algebra algebra_t, unsigned int D,
              typename measurement_backend_t>
    TRACCC_HOST_DEVICE static constexpr detray::dmatrix<algebra_t, D, 1>
    calibrated_measurement_position(
        const edm::measurement<measurement_backend_t>& measurement,
        const config& /*cfg*/) {

        // Measurement data on surface
        detray::dmatrix<algebra_t, D, 1> meas_local;
        edm::get_measurement_local<algebra_t>(measurement, meas_local);

        TRACCC_DEBUG_HOST(
            "Measurement position (uncalibrated): " << meas_local);

        assert((measurement.dimensions() > 1) ||
               (getter::element(meas_local, 1u, 0u) == 0.f));

        return meas_local;
    }

    /// Get the calibrated measurement covariance
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param measurement the measurement
    /// @param is_line whether the measurement belong to a line surface
    /// @param cfg how to apply calibrations
    ///
    /// @returns the projection matrix H
    template <detray::concepts::algebra algebra_t, unsigned int D,
              typename measurement_backend_t>
    TRACCC_HOST_DEVICE static constexpr detray::dmatrix<algebra_t, D, D>
    calibrated_measurement_covariance(
        const edm::measurement<measurement_backend_t>& measurement,
        const config& /*cfg*/) {

        // Measurement covariance
        detray::dmatrix<algebra_t, D, D> V;
        edm::get_measurement_covariance<algebra_t>(measurement, V);

        if (measurement.dimensions() == 1) {
            getter::element(V, 1u, 1u) =
                std::numeric_limits<detray::dscalar<algebra_t>>::max();
        }

        TRACCC_DEBUG_HOST("Measurement covariance (uncalibrated):\n" << V);

        return V;
    }

    /// Caluculate the predicted chi2
    ///
    /// @brief Based on "Application of Kalman filtering to track and vertex
    /// fitting", R.Frühwirth, NIM A
    ///
    /// @param measurement the measurement
    /// @param bound_params bound track parameters (state vector, covariance)
    /// @param is_line whether the measurement belong to a line surface
    ///
    /// @returns the predicted chi2
    template <typename measurement_backend_t,
              detray::concepts::algebra algebra_t>
    TRACCC_HOST_DEVICE static constexpr detray::dscalar<algebra_t>
    predicted_chi2(const edm::measurement<measurement_backend_t>& measurement,
                   const bound_track_parameters<algebra_t>& bound_params,
                   const config& cfg, const bool is_line) {

        // Measurement maximal dimension
        constexpr unsigned int D = 2;

        TRACCC_VERBOSE_HOST_DEVICE("In measurement selector...");
        TRACCC_VERBOSE_HOST_DEVICE("Measurement dim: %d",
                                   measurement.dimensions());

        assert(measurement.dimensions() == 1u ||
               measurement.dimensions() == 2u);

        TRACCC_DEBUG_HOST("Predicted param.: " << bound_params);

        assert(!bound_params.is_invalid());
        assert(!bound_params.surface_link().is_invalid());

        // Predicted vector and covariance of bound track parameters
        const traccc::bound_vector<algebra_t>& predicted_vec =
            bound_params.vector();
        const traccc::bound_matrix<algebra_t>& predicted_cov =
            bound_params.covariance();

        const matrix_t<algebra_t, D, 1> meas_local =
            calibrated_measurement_position<algebra_t, D>(measurement, cfg);

        // Measurement covariance
        const matrix_t<algebra_t, D, D> V =
            calibrated_measurement_covariance<algebra_t, D>(measurement, cfg);

        // Get the projection matrix for this measurement (obeservation model)
        const matrix_t<algebra_t, D, e_bound_size> H =
            observation_model<algebra_t, D>(measurement, bound_params, is_line);

        // Project the predicted covariance to the observation
        const matrix_t<algebra_t, D, D> R =
            H * algebra::matrix::transposed_product<false, true>(predicted_cov,
                                                                 H) +
            V;

        TRACCC_DEBUG_HOST("R:\n" << R);
        TRACCC_DEBUG_HOST_DEVICE("det(R): %f", matrix::determinant(R));
        TRACCC_DEBUG_HOST("R_inv:\n" << matrix::inverse(R));

        // Residual between measurement and (projected) vector (innovation)
        const matrix_t<algebra_t, D, 1> residual =
            meas_local - H * predicted_vec;

        TRACCC_VERBOSE_HOST("Predicted residual: " << residual);

        const matrix_t<algebra_t, 1, 1> chi2_mat =
            algebra::matrix::transposed_product<true, false>(
                residual, matrix::inverse(R)) *
            residual;

        TRACCC_VERBOSE_HOST_DEVICE("Chi2: %f", getter::element(chi2_mat, 0, 0));

        return getter::element(chi2_mat, 0, 0);
    }
};

}  // namespace traccc
