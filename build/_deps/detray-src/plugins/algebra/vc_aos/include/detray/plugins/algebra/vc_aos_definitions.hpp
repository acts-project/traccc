/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Algebra-Plugins include
#include "algebra/vc_cmath.hpp"

#define ALGEBRA_PLUGIN detray::vc_cmath

namespace detray {

/// Define scalar type
using scalar = DETRAY_CUSTOM_SCALARTYPE;

/// Define affine transformation types
/// @{
template <typename V = DETRAY_CUSTOM_SCALARTYPE>
struct vc_cmath {
    /// Define scalar type
    using value_type = V;

    template <typename T>
    using simd = T;

    using boolean = bool;
    using scalar = value_type;
    using transform3D = algebra::vc::transform3<value_type>;
    using point2D = algebra::vc::point2<value_type>;
    using point3D = algebra::vc::point3<value_type>;
    using vector3D = algebra::vc::vector3<value_type>;

    // Define matrix/vector operator
    using matrix_operator = algebra::matrix::actor<
        value_type, algebra::matrix::determinant::preset0<value_type>,
        algebra::matrix::inverse::preset0<value_type>>;
};
/// @}

// Define namespace(s)
namespace matrix = algebra::matrix;

namespace vector {

using algebra::cmath::dot;
using algebra::cmath::normalize;
using algebra::vc::math::cross;
using algebra::vc::math::dot;
using algebra::vc::math::normalize;

}  // namespace vector

namespace getter {

using algebra::cmath::eta;
using algebra::cmath::norm;
using algebra::cmath::perp;
using algebra::cmath::phi;
using algebra::cmath::theta;

using algebra::vc::math::eta;
using algebra::vc::math::norm;
using algebra::vc::math::perp;
using algebra::vc::math::phi;
using algebra::vc::math::theta;

using algebra::cmath::element;

/// Function extracting a slice from the matrix used by
/// @c algebra::vc::transform3
template <std::size_t SIZE, std::size_t ROWS, std::size_t COLS,
          typename scalar_t>
ALGEBRA_HOST_DEVICE inline Vc::array<scalar_t, SIZE> vector(
    const algebra::vc::matrix_type<scalar_t, ROWS, COLS>& m, std::size_t row,
    std::size_t col) {

    return algebra::cmath::vector_getter<std::size_t, Vc::array, scalar_t, SIZE,
                                         Vc::array<scalar_t, SIZE>>()(m, row,
                                                                      col);
}

}  // namespace getter

}  // namespace detray
