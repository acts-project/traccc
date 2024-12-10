/** Detray library, part of the ACTS project
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/track_parametrization.hpp"
#include "detray/geometry/coordinates/concentric_cylindrical2D.hpp"
#include "detray/geometry/coordinates/cylindrical2D.hpp"
#include "detray/propagator/detail/jacobian.hpp"

namespace detray::detail {

/// @brief Specialization for 2D cylindrical frames
template <typename algebra_t>
struct jacobian<cylindrical2D<algebra_t>> {

    /// @name Type definitions for the struct
    /// @{
    using coordinate_frame = cylindrical2D<algebra_t>;

    using algebra_type = algebra_t;
    using transform3_type = dtransform3D<algebra_t>;
    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;

    // Matrix operator
    using matrix_operator = dmatrix_operator<algebra_t>;
    // Rotation Matrix
    using rotation_matrix = dmatrix<algebra_t, 3, 3>;

    using bound_to_free_matrix_type = bound_to_free_matrix<algebra_t>;
    using free_to_bound_matrix_type = free_to_bound_matrix<algebra_t>;
    using free_to_path_matrix_type = free_to_path_matrix<algebra_t>;

    DETRAY_HOST_DEVICE
    static inline rotation_matrix reference_frame(const transform3_type &trf3,
                                                  const point3_type &pos,
                                                  const vector3_type &dir) {

        rotation_matrix rot = matrix_operator().template zero<3, 3>();

        // y axis of the new frame is the z axis of cylindrical coordinate
        const auto new_yaxis =
            matrix_operator().template block<3, 1>(trf3.matrix(), 0u, 2u);

        // z axis of the new frame is the vector normal to the cylinder surface
        const point3_type local =
            coordinate_frame::global_to_local_3D(trf3, pos, dir);
        const vector3_type new_zaxis = coordinate_frame::normal(trf3, local);

        // x axis
        const vector3_type new_xaxis = vector::cross(new_yaxis, new_zaxis);

        matrix_operator().element(rot, 0u, 0u) = new_xaxis[0];
        matrix_operator().element(rot, 1u, 0u) = new_xaxis[1];
        matrix_operator().element(rot, 2u, 0u) = new_xaxis[2];
        matrix_operator().template set_block<3, 1>(rot, new_yaxis, 0u, 1u);
        matrix_operator().element(rot, 0u, 2u) = new_zaxis[0];
        matrix_operator().element(rot, 1u, 2u) = new_zaxis[1];
        matrix_operator().element(rot, 2u, 2u) = new_zaxis[2];

        return rot;
    }

    DETRAY_HOST_DEVICE static inline free_to_path_matrix_type path_derivative(
        const transform3_type &trf3, const point3_type &pos,
        const vector3_type &dir, const vector3_type & /*dtds*/) {

        free_to_path_matrix_type derivative =
            matrix_operator().template zero<1u, e_free_size>();

        const point3_type local =
            coordinate_frame::global_to_local_3D(trf3, pos, dir);
        const vector3_type normal = coordinate_frame::normal(trf3, local);

        const vector3_type pos_term = -1.f / vector::dot(normal, dir) * normal;

        matrix_operator().element(derivative, 0u, e_free_pos0) = pos_term[0];
        matrix_operator().element(derivative, 0u, e_free_pos1) = pos_term[1];
        matrix_operator().element(derivative, 0u, e_free_pos2) = pos_term[2];

        return derivative;
    }

    DETRAY_HOST_DEVICE
    static inline void set_bound_pos_to_free_pos_derivative(
        bound_to_free_matrix_type &bound_to_free_jacobian,
        const transform3_type &trf3, const point3_type &pos,
        const vector3_type &dir) {

        const auto frame = reference_frame(trf3, pos, dir);

        // Get d(x,y,z)/d(loc0, loc1)
        const auto bound_pos_to_free_pos_derivative =
            matrix_operator().template block<3, 2>(frame, 0u, 0u);

        matrix_operator().set_block(bound_to_free_jacobian,
                                    bound_pos_to_free_pos_derivative,
                                    e_free_pos0, e_bound_loc0);
    }

    DETRAY_HOST_DEVICE
    static inline void set_free_pos_to_bound_pos_derivative(
        free_to_bound_matrix_type &free_to_bound_jacobian,
        const transform3_type &trf3, const point3_type &pos,
        const vector3_type &dir) {

        const auto frame = reference_frame(trf3, pos, dir);
        const auto frameT = matrix_operator().transpose(frame);

        // Get d(loc0, loc1)/d(x,y,z)
        const auto free_pos_to_bound_pos_derivative =
            matrix_operator().template block<2, 3>(frameT, 0u, 0u);

        matrix_operator().set_block(free_to_bound_jacobian,
                                    free_pos_to_bound_pos_derivative,
                                    e_bound_loc0, e_free_pos0);
    }

    DETRAY_HOST_DEVICE
    static inline void set_bound_angle_to_free_pos_derivative(
        bound_to_free_matrix_type & /*bound_to_free_jacobian*/,
        const transform3_type & /*trf3*/, const point3_type & /*pos*/,
        const vector3_type & /*dir*/) {
        // Do nothing
    }
};

template <typename algebra_t>
struct jacobian<concentric_cylindrical2D<algebra_t>>
    : public jacobian<cylindrical2D<algebra_t>> {};

}  // namespace detray::detail
