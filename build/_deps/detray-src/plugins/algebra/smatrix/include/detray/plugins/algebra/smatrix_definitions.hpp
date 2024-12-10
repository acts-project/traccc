/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// TODO: Remove this when gcc fixes their false positives.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

// Algebra-Plugins include
#include "algebra/smatrix_smatrix.hpp"

#define ALGEBRA_PLUGIN detray::smatrix

namespace detray {

// Define scalar type
using scalar = DETRAY_CUSTOM_SCALARTYPE;

/// Define affine transformation types
/// @{
template <typename V = DETRAY_CUSTOM_SCALARTYPE>
struct smatrix {
    /// Define scalar type
    using value_type = V;

    template <typename T>
    using simd = T;

    using boolean = bool;
    using scalar = value_type;
    using transform3D = algebra::smatrix::transform3<value_type>;
    using point2D = algebra::smatrix::point2<value_type>;
    using point3D = algebra::smatrix::point3<value_type>;
    using vector3D = algebra::smatrix::vector3<value_type>;

    // Define matrix/vector operator
    using matrix_operator = algebra::matrix::actor<value_type>;
};
/// @}

// Define namespace(s)
namespace matrix = algebra::matrix;

namespace vector {

using algebra::smatrix::math::cross;
using algebra::smatrix::math::dot;
using algebra::smatrix::math::normalize;

}  // namespace vector

namespace getter {

using algebra::smatrix::math::eta;
using algebra::smatrix::math::norm;
using algebra::smatrix::math::perp;
using algebra::smatrix::math::phi;
using algebra::smatrix::math::theta;

using algebra::smatrix::math::element;

/// Function extracting a slice from the matrix used by
/// @c algebra::smatrix::transform3
template <unsigned int SIZE, unsigned int ROWS, unsigned int COLS,
          typename scalar_t>
ALGEBRA_HOST_DEVICE inline auto vector(
    const ROOT::Math::SMatrix<scalar_t, ROWS, COLS>& m, unsigned int row,
    unsigned int col) {

    return m.template SubCol<algebra::smatrix::storage_type<scalar_t, SIZE>>(
        col, row);
}

}  // namespace getter

}  // namespace detray
