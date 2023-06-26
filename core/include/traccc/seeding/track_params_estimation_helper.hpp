/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"

// System include(s).
#include <cmath>

namespace traccc {

/// helper functions (for both cpu and gpu) to perform conformal transformation
///
/// @param x is the x value
/// @param y is the y value
/// @return is the conformal transformation result
inline TRACCC_HOST_DEVICE vector2 uv_transform(const scalar& x,
                                               const scalar& y) {
    vector2 uv;
    scalar denominator = x * x + y * y;
    uv[0] = x / denominator;
    uv[1] = y / denominator;
    return uv;
}

/// helper functions (for both cpu and gpu) to calculate bound track parameter
/// at the bottom spacepoint
///
/// @param seed is the input seed
/// @param bfield is the magnetic field
/// @param mass is the mass of particle
template <typename spacepoint_collection_t>
inline TRACCC_HOST_DEVICE bound_vector seed_to_bound_vector(
    const spacepoint_collection_t& sp_collection, const seed& seed,
    const vector3& bfield, const scalar mass) {

    bound_vector params;

    const auto& spB = sp_collection.at(seed.spB_link);
    const auto& spM = sp_collection.at(seed.spM_link);
    const auto& spT = sp_collection.at(seed.spT_link);

    darray<vector3, 3> sp_global_positions;
    sp_global_positions[0] = spB.global;
    sp_global_positions[1] = spM.global;
    sp_global_positions[2] = spT.global;

    // Define a new coordinate frame with its origin at the bottom space
    // point, z axis long the magnetic field direction and y axis
    // perpendicular to vector from the bottom to middle space point.
    // Hence, the projection of the middle space point on the tranverse
    // plane will be located at the x axis of the new frame.
    vector3 relVec = sp_global_positions[1] - sp_global_positions[0];
    vector3 newZAxis = vector::normalize(bfield);
    vector3 newYAxis = vector::normalize(vector::cross(newZAxis, relVec));
    vector3 newXAxis = vector::cross(newYAxis, newZAxis);

    // The center of the new frame is at the bottom space point
    vector3 translation = sp_global_positions[0];

    transform3 trans(translation, newZAxis, newXAxis);

    // The coordinate of the middle and top space point in the new frame
    auto local1 = trans.point_to_local(sp_global_positions[1]);
    auto local2 = trans.point_to_local(sp_global_positions[2]);

    // The uv1.y() should be zero
    vector2 uv1 = uv_transform(local1[0], local1[1]);
    vector2 uv2 = uv_transform(local2[0], local2[1]);

    // A,B are slope and intercept of the straight line in the u,v plane
    // connecting the three points
    scalar A = (uv2[1] - uv1[1]) / (uv2[0] - uv1[0]);
    scalar B = uv2[1] - A * uv2[0];

    // Curvature (with a sign) estimate
    scalar rho = -2.0f * B / getter::perp(vector2{1., A});
    // The projection of the top space point on the transverse plane of
    // the new frame
    scalar rn = local2[0] * local2[0] + local2[1] * local2[1];
    // The (1/tanTheta) of momentum in the new frame
    static constexpr scalar G = static_cast<scalar>(1.f / 24.f);
    scalar invTanTheta =
        local2[2] * std::sqrt(1.f / rn) / (1.f + G * rho * rho * rn);

    // The momentum direction in the new frame (the center of the circle
    // has the coordinate (-1.*A/(2*B), 1./(2*B)))
    vector3 transDirection =
        vector3({1., A, scalar(getter::perp(vector2{1., A})) * invTanTheta});
    // Transform it back to the original frame
    vector3 direction =
        transform3::rotate(trans._data, vector::normalize(transDirection));

    // The estimated phi and theta
    getter::element(params, e_bound_phi, 0) = getter::phi(direction);
    getter::element(params, e_bound_theta, 0) = getter::theta(direction);

    // The measured loc0 and loc1
    const auto& meas_for_spB = spB.meas;
    getter::element(params, e_bound_loc0, 0) = meas_for_spB.local[0];
    getter::element(params, e_bound_loc1, 0) = meas_for_spB.local[1];

    // The estimated q/pt in [GeV/c]^-1 (note that the pt is the
    // projection of momentum on the transverse plane of the new frame)
    scalar qOverPt = rho / getter::norm(bfield);
    // The estimated q/p in [GeV/c]^-1
    getter::element(params, e_bound_qoverp, 0) =
        qOverPt / getter::perp(vector2{1., invTanTheta});

    // The estimated momentum, and its projection along the magnetic
    // field diretion
    scalar pInGeV = std::abs(1.0f / getter::element(params, e_bound_qoverp, 0));
    scalar pzInGeV = 1.0f / std::abs(qOverPt) * invTanTheta;
    scalar massInGeV = mass / unit<scalar>::GeV;

    // The estimated velocity, and its projection along the magnetic
    // field diretion
    scalar v = pInGeV / getter::perp(vector2{pInGeV, massInGeV});
    scalar vz = pzInGeV / getter::perp(vector2{pInGeV, massInGeV});
    // The z coordinate of the bottom space point along the magnetic
    // field direction
    scalar pathz =
        vector::dot(sp_global_positions[0], bfield) / getter::norm(bfield);

    // The estimated time (use path length along magnetic field only if
    // it's not zero)
    if (pathz != 0) {
        getter::element(params, e_bound_time, 0) = pathz / vz;
    } else {
        getter::element(params, e_bound_time, 0) =
            getter::norm(sp_global_positions[0]) / v;
    }

    return params;
}

}  // namespace traccc
