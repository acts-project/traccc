/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
#include "algebra/qualifiers.hpp"

// System include(s).
#include <type_traits>
#include <utility>

namespace algebra::vc::math {

/** Dot product between two input vectors
 *
 * @tparam vector_type generic input vector type
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename vector_type1, typename vector_type2,
          std::enable_if_t<
              std::is_class<decltype(std::declval<vector_type1>() *
                                     std::declval<vector_type2>())>::value,
              bool> = true>
ALGEBRA_HOST_DEVICE inline auto dot(const vector_type1 &a,
                                    const vector_type2 &b) {

  return (a * b).sum();
}

/** Get a normalized version of the input vector
 *
 * @tparam vector_type generic input vector type
 *
 * @param v the input vector
 **/
template <typename vector_type>
ALGEBRA_HOST_DEVICE inline auto normalize(const vector_type &v) {

  return v / algebra::math::sqrt(dot(v, v));
}

/** Cross product between two input vectors - 3 Dim
 *
 * @tparam vector_type generic input vector type
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector representing the cross product
 **/
template <typename vector_type1, typename vector_type2,
          std::enable_if_t<std::is_fundamental<decltype(
                               std::declval<vector_type1>()[0] *
                               std::declval<vector_type2>()[0])>::value,
                           bool> = true>
ALGEBRA_HOST_DEVICE inline auto cross(const vector_type1 &a,
                                      const vector_type2 &b)
    -> decltype(a * b - a * b) {

  return {a[1] * b[2] - b[1] * a[2], a[2] * b[0] - b[2] * a[0],
          a[0] * b[1] - b[0] * a[1], 0};
}

}  // namespace algebra::vc::math
