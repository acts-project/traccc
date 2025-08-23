/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc::edm {

/// Get the local position of a measurement as a matrix
///
/// @tparam algebra_t The algebra type used to describe the tracks
/// @tparam size_t The type of the matrix size variable
/// @tparam D The dimension of the matrix
///
/// @param meas The measurement to extract the local position from
/// @param pos The matrix to fill with the local position of the measurement
///
template <detray::concepts::algebra algebra_t, std::integral size_t, size_t D>
TRACCC_HOST_DEVICE void get_measurement_local(
    const measurement& meas, detray::dmatrix<algebra_t, D, 1>& pos);

/// Get the covariance of a measurement as a matrix
///
/// @tparam algebra_t The algebra type used to describe the tracks
/// @tparam size_t The type of the matrix size variable
/// @tparam D The dimension of the matrix
///
/// @param meas The measurement to extract the covariance from
/// @param cov The matrix to fill with the covariance of the measurement
///
template <detray::concepts::algebra algebra_t, std::integral size_t, size_t D>
TRACCC_HOST_DEVICE void get_measurement_covariance(
    const measurement& meas, detray::dmatrix<algebra_t, D, D>& cov);

}  // namespace traccc::edm

// Include the implementation.
#include "traccc/edm/impl/measurement_helpers.ipp"
