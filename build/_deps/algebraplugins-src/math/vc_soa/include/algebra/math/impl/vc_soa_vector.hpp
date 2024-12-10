/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
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

namespace algebra::vc_soa::math {

/// Cross product between two input vectors - 3 Dim
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param a the first input vector
/// @param b the second input vector
///
/// @return a vector (expression) representing the cross product
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t,
          std::enable_if_t<N == 3, bool> = true>
ALGEBRA_HOST_DEVICE inline storage::vector<N, Vc::Vector<scalar_t>, array_t>
cross(const storage::vector<N, Vc::Vector<scalar_t>, array_t> &a,
      const storage::vector<N, Vc::Vector<scalar_t>, array_t> &b) {

  return {Vc::fma(a[1], b[2], -b[1] * a[2]), Vc::fma(a[2], b[0], -b[2] * a[0]),
          Vc::fma(a[0], b[1], -b[0] * a[1])};
}

/// Dot product between two input vectors
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param a the first input vector
/// @param b the second input vector
///
/// @return the scalar dot product value
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t>
ALGEBRA_HOST_DEVICE inline Vc::Vector<scalar_t> dot(
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &a,
    const storage::vector<N, Vc::Vector<scalar_t>, array_t> &b) {

  auto ret = Vc::Vector<scalar_t>::Zero();
  for (unsigned int i{0u}; i < N; i++) {
    ret = Vc::fma(a[i], b[i], ret);
  }
  return ret;
}

/// Get a normalized version of the input vector
///
/// @tparam N dimension of the vector
/// @tparam scalar_t scalar type
/// @tparam array_t array type that holds the vector elements
///
/// @param v the input vector
template <std::size_t N, typename scalar_t,
          template <typename, std::size_t> class array_t>
ALGEBRA_HOST_DEVICE inline storage::vector<N, Vc::Vector<scalar_t>, array_t>
normalize(const storage::vector<N, Vc::Vector<scalar_t>, array_t> &v) {

  return (Vc::Vector<scalar_t>::One() / Vc::sqrt(dot(v, v))) * v;

  // Less accurate, but faster
  // return Vc::reciprocal(Vc::sqrt(dot(v, v))) * v;

  // Even less accurate, but even faster
  // return Vc::rsqrt(dot(v, v)) * v;
}

}  // namespace algebra::vc_soa::math
