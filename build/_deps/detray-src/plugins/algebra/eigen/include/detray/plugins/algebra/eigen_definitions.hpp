/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Algebra-Plugins include
#include "algebra/eigen_eigen.hpp"

#define ALGEBRA_PLUGIN detray::eigen

namespace detray {

// Define scalar type
using scalar = DETRAY_CUSTOM_SCALARTYPE;

/// Define affine transformation types
/// @{
template <typename V = DETRAY_CUSTOM_SCALARTYPE>
struct eigen {
    /// Define scalar type
    using value_type = V;

    template <typename T>
    using simd = T;

    using boolean = bool;
    using scalar = value_type;
    using transform3D = algebra::eigen::transform3<value_type>;
    using point2D = algebra::eigen::point2<value_type>;
    using point3D = algebra::eigen::point3<value_type>;
    using vector3D = algebra::eigen::vector3<value_type>;

    // Define matrix/vector operator
    using matrix_operator = algebra::matrix::actor<value_type>;
};
/// @}

// Define namespace(s)
namespace matrix = algebra::matrix;

namespace vector {

using algebra::eigen::math::cross;
using algebra::eigen::math::dot;
using algebra::eigen::math::normalize;

}  // namespace vector

namespace getter {

using algebra::eigen::math::eta;
using algebra::eigen::math::norm;
using algebra::eigen::math::perp;
using algebra::eigen::math::phi;
using algebra::eigen::math::theta;

using algebra::eigen::math::element;

/// Function extracting a slice from the matrix used by
/// @c algebra::eigen::transform3
template <unsigned int SIZE, typename derived_type>
ALGEBRA_HOST_DEVICE inline auto vector(const Eigen::MatrixBase<derived_type>& m,
                                       std::size_t row, std::size_t col) {

    return m.template block<SIZE, 1>(static_cast<Eigen::Index>(row),
                                     static_cast<Eigen::Index>(col));
}

}  // namespace getter

}  // namespace detray
