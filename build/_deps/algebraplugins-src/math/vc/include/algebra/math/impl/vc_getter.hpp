/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/common.hpp"
#include "algebra/math/impl/vc_vector.hpp"
#include "algebra/qualifiers.hpp"

namespace algebra::vc::math {

/** This method retrieves phi from a vector, vector base with rows >= 2
 *
 * @param v the input vector
 **/
template <typename vector_type>
ALGEBRA_HOST_DEVICE inline auto phi(const vector_type &v) noexcept {

  return algebra::math::atan2(v[1], v[0]);
}

/** This method retrieves theta from a vector, vector base with rows >= 3
 *
 * @param v the input vector
 **/
template <typename vector_type>
ALGEBRA_HOST_DEVICE inline auto theta(const vector_type &v) noexcept {

  return algebra::math::atan2(algebra::math::sqrt(v[0] * v[0] + v[1] * v[1]),
                              v[2]);
}

/** This method retrieves the perpendicular magnitude of a vector with rows >= 2
 *
 * @param v the input vector
 **/
template <typename vector_type>
ALGEBRA_HOST_DEVICE inline auto perp(const vector_type &v) noexcept {

  return algebra::math::sqrt(v[0] * v[0] + v[1] * v[1]);
}

/** This method retrieves the norm of a vector, no dimension restriction
 *
 * @param v the input vector
 **/
template <typename vector_type>
ALGEBRA_HOST_DEVICE inline auto norm(const vector_type &v) {

  return algebra::math::sqrt(dot(v, v));
}

/** This method retrieves the pseudo-rapidity from a vector or vector base with
 * rows >= 3
 *
 * @param v the input vector
 **/
template <typename vector_type>
ALGEBRA_HOST_DEVICE inline auto eta(const vector_type &v) noexcept {

  return algebra::math::atanh(v[2] / norm(v));
}

}  // namespace algebra::vc::math
