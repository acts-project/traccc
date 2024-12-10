/** Detray library, part of the ACTS project
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/track_parametrization.hpp"
#include "detray/geometry/coordinates/polar2D.hpp"
#include "detray/propagator/detail/jacobian.hpp"

namespace detray::detail {

/// @brief Specialization for 2D cartesian frames
template <typename algebra_t>
struct jacobian<polar2D<algebra_t>> {

    /// @name Type definitions for the struct
    /// @{
    using coordinate_frame = polar2D<algebra_t>;

    using algebra_type = algebra_t;
    using transform3_type = dtransform3D<algebra_t>;
    using scalar_type = dscalar<algebra_t>;
    using point2_type = dpoint2D<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;

    // Matrix operator
    using matrix_operator = dmatrix_operator<algebra_t>;
    // 2D matrix type
    template <std::size_t ROWS, std::size_t COLS>
    using matrix_type = dmatrix<algebra_t, ROWS, COLS>;
    // Rotation Matrix
    using rotation_matrix = matrix_type<3, 3>;

    using bound_to_free_matrix_type = bound_to_free_matrix<algebra_t>;
    using free_to_bound_matrix_type = free_to_bound_matrix<algebra_t>;
    using free_to_path_matrix_type = free_to_path_matrix<algebra_t>;
    /// @}

    DETRAY_HOST_DEVICE
    static inline rotation_matrix reference_frame(
        const transform3_type &trf3, const point3_type & /*pos*/,
        const vector3_type & /*dir*/) {
        return trf3.rotation();
    }

    DETRAY_HOST_DEVICE static inline free_to_path_matrix_type path_derivative(
        const transform3_type &trf3, const point3_type & /*pos*/,
        const vector3_type &dir, const vector3_type & /*dtds*/) {

        free_to_path_matrix_type derivative =
            matrix_operator().template zero<1u, e_free_size>();

        const vector3_type normal = coordinate_frame::normal(trf3);

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

        matrix_type<3, 2> bound_pos_to_free_pos_derivative =
            matrix_operator().template zero<3, 2>();

        const point2_type local =
            coordinate_frame::global_to_local(trf3, pos, dir);
        const scalar_type lrad{local[0]};
        const scalar_type lphi{local[1]};

        const scalar_type lcos_phi{math::cos(lphi)};
        const scalar_type lsin_phi{math::sin(lphi)};

        // reference matrix
        const auto frame = reference_frame(trf3, pos, dir);

        // dxdu = d(x,y,z)/du
        const matrix_type<3, 1> dxdL =
            matrix_operator().template block<3, 1>(frame, 0u, 0u);
        // dxdv = d(x,y,z)/dv
        const matrix_type<3, 1> dydL =
            matrix_operator().template block<3, 1>(frame, 0u, 1u);

        const matrix_type<3, 1> col0 = dxdL * lcos_phi + dydL * lsin_phi;
        const matrix_type<3, 1> col1 =
            (dydL * lcos_phi - dxdL * lsin_phi) * lrad;

        matrix_operator().template set_block<3, 1>(
            bound_pos_to_free_pos_derivative, col0, e_free_pos0, e_bound_loc0);
        matrix_operator().template set_block<3, 1>(
            bound_pos_to_free_pos_derivative, col1, e_free_pos0, e_bound_loc1);

        matrix_operator().set_block(bound_to_free_jacobian,
                                    bound_pos_to_free_pos_derivative,
                                    e_free_pos0, e_bound_loc0);
    }

    DETRAY_HOST_DEVICE
    static inline void set_free_pos_to_bound_pos_derivative(
        free_to_bound_matrix_type &free_to_bound_jacobian,
        const transform3_type &trf3, const point3_type &pos,
        const vector3_type &dir) {

        matrix_type<2, 3> free_pos_to_bound_pos_derivative =
            matrix_operator().template zero<2, 3>();

        const point2_type local =
            coordinate_frame::global_to_local(trf3, pos, dir);

        const scalar_type lrad{local[0]};
        const scalar_type lphi{local[1]};

        const scalar_type lcos_phi{math::cos(lphi)};
        const scalar_type lsin_phi{math::sin(lphi)};

        // reference matrix
        const auto frame = reference_frame(trf3, pos, dir);
        const auto frameT = matrix_operator().transpose(frame);

        // dudG = du/d(x,y,z)
        const matrix_type<1, 3> dudG =
            matrix_operator().template block<1, 3>(frameT, 0u, 0u);
        // dvdG = dv/d(x,y,z)
        const matrix_type<1, 3> dvdG =
            matrix_operator().template block<1, 3>(frameT, 1u, 0u);

        const matrix_type<1, 3> row0 = dudG * lcos_phi + dvdG * lsin_phi;
        const matrix_type<1, 3> row1 =
            1.f / lrad * (lcos_phi * dvdG - lsin_phi * dudG);

        matrix_operator().template set_block<1, 3>(
            free_pos_to_bound_pos_derivative, row0, e_bound_loc0, e_free_pos0);
        matrix_operator().template set_block<1, 3>(
            free_pos_to_bound_pos_derivative, row1, e_bound_loc1, e_free_pos0);

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

}  // namespace detray::detail
