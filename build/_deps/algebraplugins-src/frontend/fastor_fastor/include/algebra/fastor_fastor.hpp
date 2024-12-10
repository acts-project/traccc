/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/cmath.hpp"
#include "algebra/math/fastor.hpp"
#include "algebra/storage/fastor.hpp"

// Fastor include(s).
#ifdef _MSC_VER
#pragma warning(disable : 4244 4701 4702)
#endif  // MSVC
#include <Fastor/Fastor.h>
#ifdef _MSC_VER
#pragma warning(default : 4244 4701 4702)
#endif  // MSVC

namespace algebra {

namespace getter {

/// @name Getter functions on @c algebra::fastor::storage_type
/// @{

using fastor::math::eta;
using fastor::math::norm;
using fastor::math::perp;
using fastor::math::phi;
using fastor::math::theta;

/// @}

/// Function extracting a slice from the matrix used by
/// @c algebra::fastor::transform3
template <std::size_t SIZE, std::size_t ROWS, std::size_t COLS,
          typename scalar_t>
ALGEBRA_HOST_DEVICE inline auto vector(
    const Fastor::Tensor<scalar_t, ROWS, COLS>& m, std::size_t row,
    std::size_t col) {

  return fastor::storage_type<scalar_t, SIZE>(
      m(Fastor::seq(static_cast<int>(row), static_cast<int>(row + SIZE)),
        static_cast<int>(col)));
}

/// @name Getter functions on @c algebra::fastor::matrix_type
/// @{

using fastor::math::element;

/// @}

}  // namespace getter

namespace vector {

/// @name Vector functions on @c algebra::fastor::storage_type
/// @{

using fastor::math::cross;
using fastor::math::dot;
using fastor::math::normalize;

/// @}

}  // namespace vector

namespace matrix {

template <typename scalar_t>
using actor = fastor::matrix::actor<scalar_t>;

}  // namespace matrix

namespace fastor {

template <typename T>
using transform3 = math::transform3<T, matrix::actor<T>>;

}  // namespace fastor

}  // namespace algebra
