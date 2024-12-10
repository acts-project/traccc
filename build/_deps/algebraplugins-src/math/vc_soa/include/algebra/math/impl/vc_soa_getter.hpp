/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/math/impl/vc_soa_vector.hpp"
#include "algebra/qualifiers.hpp"
#include "algebra/storage/vector.hpp"

// Vc include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Vc/Vc>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

// System include(s)
#include <cmath>
#include <type_traits>

namespace algebra::vc_soa::math {

/// This method retrieves phi from a vector, vector base with rows >= 2
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param v the input vector
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t,
          std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST_DEVICE inline auto phi(
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &v) {

  return Vc::atan2(v[1], v[0]);
}

/// This method retrieves the perpenticular magnitude of a vector with rows >= 2
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param v the input vector
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t,
          std::enable_if_t<N >= 2, bool> = true>
ALGEBRA_HOST_DEVICE inline auto perp(
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &v) {

  auto tmp = Vc::Vector<scalar_t>::Zero();
  tmp = Vc::fma(v[0], v[0], tmp);
  tmp = Vc::fma(v[1], v[1], tmp);

  return Vc::sqrt(tmp);
}

/// This method retrieves theta from a vector, vector base with rows >= 3
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param v the input vector
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t,
          std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST_DEVICE inline auto theta(
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &v) {

  return Vc::atan2(perp(v), v[2]);
}

/// This method retrieves the norm of a vector, no dimension restriction
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param v the input vector
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t>
ALGEBRA_HOST_DEVICE inline auto norm(
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &v) {

  return Vc::sqrt(dot(v, v));
}

/// This method retrieves the pseudo-rapidity from a vector or vector base with
/// rows >= 3
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param v the input vector
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t,
          std::enable_if_t<N >= 3, bool> = true>
ALGEBRA_HOST_DEVICE inline auto eta(
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &v) {

  // atanh does not exist in Vc
  auto atanh_func = [](scalar_t e) { return std::atanh(e); };

  return (v[2] / norm(v)).apply(atanh_func);

  // Faster, but less accurate
  // return (Vc::reciprocal(norm(v)) * v[2]).apply(atanh_func);

  // Even faster, but even less accurate
  // return (Vc::rsqrt(dot(v, v)) * v[2]).apply(atanh_func);
}

}  // namespace algebra::vc_soa::math
