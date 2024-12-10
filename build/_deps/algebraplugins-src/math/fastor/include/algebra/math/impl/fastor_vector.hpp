/** Algebra plugins, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// Fastor include(s).
#ifdef _MSC_VER
#pragma warning(disable : 4244 4701 4702)
#endif  // MSVC
#include <Fastor/Fastor.h>
#ifdef _MSC_VER
#pragma warning(default : 4244 4701 4702)
#endif  // MSVC

namespace algebra::fastor::math {

/** Get a normalized version of the input vector
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline Fastor::Tensor<scalar_t, N> normalize(
    const Fastor::Tensor<scalar_t, N> &v) {

  return (static_cast<scalar_t>(1.0) / Fastor::norm(v)) * v;
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST_DEVICE inline scalar_t dot(const Fastor::Tensor<scalar_t, N> &a,
                                        const Fastor::Tensor<scalar_t, N> &b) {
  return Fastor::inner(a, b);
}

/** Dot product between Tensor<scalar_t, N> and Tensor<scalar_t, N, 1>
 *
 * @param a the first input vector
 * @param b the second input Tensor<scalar_t, N, 1>
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t dot(const Fastor::Tensor<scalar_t, N> &a,
                                 const Fastor::Tensor<scalar_t, N, 1> &b) {

  // We need to specify the type of the Tensor slice because Fastor by default
  // is lazy, so it returns an intermediate type which does not play well with
  // the Fastor::inner function.
  return Fastor::inner(a,
                       Fastor::Tensor<scalar_t, N>(b(Fastor::fseq<0, N>(), 0)));
}

/** Dot product between Tensor<scalar_t, N> and Tensor<scalar_t, N, 1>
 *
 * @param a the second input Tensor<scalar_t, N, 1>
 * @param b the first input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t dot(const Fastor::Tensor<scalar_t, N, 1> &a,
                                 const Fastor::Tensor<scalar_t, N> &b) {

  return Fastor::inner(Fastor::Tensor<scalar_t, N>(a(Fastor::fseq<0, N>(), 0)),
                       b);
}

/** Dot product between two Tensor<scalar_t, 3, 1>
 *
 * @param a the second input Tensor<scalar_t, 3, 1>
 * @param b the first input Tensor<scalar_t, 3, 1>
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t dot(const Fastor::Tensor<scalar_t, N, 1> &a,
                                 const Fastor::Tensor<scalar_t, N, 1> &b) {

  return Fastor::inner(Fastor::Tensor<scalar_t, N>(a(Fastor::fseq<0, 3>(), 0)),
                       Fastor::Tensor<scalar_t, N>(b(Fastor::fseq<0, 3>(), 0)));
}

/** Cross product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST_DEVICE inline Fastor::Tensor<scalar_t, 3> cross(
    const Fastor::Tensor<scalar_t, 3> &a,
    const Fastor::Tensor<scalar_t, 3> &b) {
  return Fastor::cross(a, b);
}

/** Cross product between Tensor<scalar_t, 3> and Tensor<scalar_t, 3, 1>
 *
 * @param a the first input vector
 * @param b the second input Tensor<scalar_t, 3, 1>
 *
 * @return a vector representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline Fastor::Tensor<scalar_t, 3> cross(
    const Fastor::Tensor<scalar_t, 3> &a,
    const Fastor::Tensor<scalar_t, 3, 1> &b) {

  // We need to specify the type of the Tensor slice because Fastor by default
  // is lazy, so it returns an intermediate type which does not play well with
  // the Fastor::cross function.
  return Fastor::cross(a,
                       Fastor::Tensor<scalar_t, 3>(b(Fastor::fseq<0, 3>(), 0)));
}

/** Cross product between Tensor<scalar_t, 3> and Tensor<scalar_t, 3, 1>
 *
 * @param a the second input Tensor<scalar_t, 3, 1>
 * @param b the first input vector
 *
 * @return a vector representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline Fastor::Tensor<scalar_t, 3> cross(
    const Fastor::Tensor<scalar_t, 3, 1> &a,
    const Fastor::Tensor<scalar_t, 3> &b) {

  return Fastor::cross(Fastor::Tensor<scalar_t, 3>(a(Fastor::fseq<0, 3>(), 0)),
                       b);
}

/** Cross product between two Tensor<scalar_t, 3, 1>
 *
 * @param a the second input Tensor<scalar_t, 3, 1>
 * @param b the first input Tensor<scalar_t, 3, 1>
 *
 * @return a vector representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline Fastor::Tensor<scalar_t, 3> cross(
    const Fastor::Tensor<scalar_t, 3, 1> &a,
    const Fastor::Tensor<scalar_t, 3, 1> &b) {

  return Fastor::cross(Fastor::Tensor<scalar_t, 3>(a(Fastor::fseq<0, 3>(), 0)),
                       Fastor::Tensor<scalar_t, 3>(b(Fastor::fseq<0, 3>(), 0)));
}

}  // namespace algebra::fastor::math
