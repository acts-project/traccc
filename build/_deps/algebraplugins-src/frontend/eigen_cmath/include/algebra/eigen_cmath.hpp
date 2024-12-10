/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/cmath.hpp"
#include "algebra/math/eigen.hpp"
#include "algebra/storage/eigen.hpp"

// Eigen include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Eigen/Geometry>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

// System include(s).
#include <type_traits>

namespace algebra {

namespace getter {

/// @name Getter functions on @c algebra::eigen::storage_type
/// @{

using eigen::math::eta;
using eigen::math::norm;
using eigen::math::perp;
using eigen::math::phi;
using eigen::math::theta;

/// @}

/// Function extracting a slice from the matrix used by
/// @c algebra::eigen::transform3
template <unsigned int SIZE, typename derived_type>
ALGEBRA_HOST_DEVICE inline auto vector(const Eigen::MatrixBase<derived_type>& m,
                                       std::size_t row, std::size_t col) {

  return m.template block<SIZE, 1>(static_cast<Eigen::Index>(row),
                                   static_cast<Eigen::Index>(col));
}

/// @name Getter functions on @c algebra::eigen::matrix_type
/// @{

using eigen::math::element;

/// @}

}  // namespace getter

namespace vector {

/// @name Vector functions on @c algebra::eigen::storage_type
/// @{

using eigen::math::cross;
using eigen::math::dot;
using eigen::math::normalize;

/// @}

}  // namespace vector

namespace matrix {

using size_type = eigen::size_type;
template <typename T, size_type N>
using array_type = eigen::storage_type<T, N>;
template <typename T, size_type ROWS, size_type COLS>
using matrix_type = eigen::matrix_type<T, ROWS, COLS>;
using element_getter = eigen::math::element_getter;
using block_getter = eigen::math::block_getter;

// matrix actor
template <typename scalar_t, typename determinant_actor_t,
          typename inverse_actor_t>
using actor = cmath::matrix::actor<size_type, array_type, matrix_type, scalar_t,
                                   determinant_actor_t, inverse_actor_t,
                                   element_getter, block_getter>;

namespace determinant {

// determinant aggregation
template <typename scalar_t, class... As>
using actor =
    cmath::matrix::determinant::actor<size_type, matrix_type, scalar_t, As...>;

// determinant::cofactor
template <typename scalar_t, size_type... Ds>
using cofactor =
    cmath::matrix::determinant::cofactor<size_type, matrix_type, scalar_t,
                                         element_getter, Ds...>;

// determinant::partial_pivot_lud
template <typename scalar_t, size_type... Ds>
using partial_pivot_lud = cmath::matrix::determinant::partial_pivot_lud<
    size_type, matrix_type, scalar_t, element_getter, Ds...>;

// determinant::hard_coded
template <typename scalar_t, size_type... Ds>
using hard_coded =
    cmath::matrix::determinant::hard_coded<size_type, matrix_type, scalar_t,
                                           element_getter, Ds...>;

// preset(s) as standard option(s) for user's convenience
template <typename scalar_t>
using preset0 =
    actor<scalar_t, partial_pivot_lud<scalar_t>, hard_coded<scalar_t, 2, 4>>;

}  // namespace determinant

namespace inverse {

// inverion aggregation
template <typename scalar_t, class... As>
using actor =
    cmath::matrix::inverse::actor<size_type, matrix_type, scalar_t, As...>;

// inverse::cofactor
template <typename scalar_t, size_type... Ds>
using cofactor =
    cmath::matrix::inverse::cofactor<size_type, matrix_type, scalar_t,
                                     element_getter, Ds...>;

// inverse::partial_pivot_lud
template <typename scalar_t, size_type... Ds>
using partial_pivot_lud =
    cmath::matrix::inverse::partial_pivot_lud<size_type, matrix_type, scalar_t,
                                              element_getter, Ds...>;

// inverse::hard_coded
template <typename scalar_t, size_type... Ds>
using hard_coded =
    cmath::matrix::inverse::hard_coded<size_type, matrix_type, scalar_t,
                                       element_getter, Ds...>;

// preset(s) as standard option(s) for user's convenience
template <typename scalar_t>
using preset0 =
    actor<scalar_t, partial_pivot_lud<scalar_t>, hard_coded<scalar_t, 2, 4>>;

}  // namespace inverse

}  // namespace matrix

namespace eigen {

/// @name cmath based transforms on @c algebra::matrix::actor
/// @{

template <typename T>
using transform3_actor =
    algebra::matrix::actor<T, algebra::matrix::determinant::preset0<T>,
                           algebra::matrix::inverse::preset0<T>>;

template <typename T>
using transform3 = cmath::transform3<transform3_actor<T>>;

/// @}

}  // namespace eigen

}  // namespace algebra
