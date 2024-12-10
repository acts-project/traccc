/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// ROOT/Smatrix include(s).
#include <Math/Expression.h>
#include <Math/Functions.h>
#include <Math/SVector.h>
#include <TMath.h>

namespace algebra::smatrix::math {

/** Get a normalized version of the input vector
 *
 * @param v the input vector
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, N> normalize(
    const ROOT::Math::SVector<scalar_t, N> &v) {

  return ROOT::Math::Unit(v);
}
/** Get a normalized version of the input vector
 *
 * @param v the input vector
 **/
template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, N> normalize(
    const ROOT::Math::VecExpr<A, scalar_t, N> &v) {

  return ROOT::Math::Unit(v);
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::SVector<scalar_t, N> &a,
                                 const ROOT::Math::SVector<scalar_t, N> &b) {

  return ROOT::Math::Dot(a, b);
}
/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::SVector<scalar_t, N> &a,
                                 const ROOT::Math::VecExpr<A, scalar_t, N> &b) {

  return ROOT::Math::Dot(a, b);
}
/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::VecExpr<A, scalar_t, N> &a,
                                 const ROOT::Math::SVector<scalar_t, N> &b) {

  return ROOT::Math::Dot(a, b);
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::VecExpr<A, scalar_t, N> &a,
                                 const ROOT::Math::VecExpr<A, scalar_t, N> &b) {

  return ROOT::Math::Dot(a, b);
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::SMatrix<scalar_t, N, 1> &a,
                                 const ROOT::Math::VecExpr<A, scalar_t, N> &b) {

  return ROOT::Math::Dot(a.Col(0), b);
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, class A, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::VecExpr<A, scalar_t, N> &a,
                                 const ROOT::Math::SMatrix<scalar_t, N, 1> &b) {
  return dot(b, a);
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::SMatrix<scalar_t, N, 1> &a,
                                 const ROOT::Math::SVector<scalar_t, N> &b) {

  return ROOT::Math::Dot(a.Col(0), b);
}

/** Dot product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return the scalar dot product value
 **/
template <typename scalar_t, auto N>
ALGEBRA_HOST inline scalar_t dot(const ROOT::Math::SVector<scalar_t, N> &a,
                                 const ROOT::Math::SMatrix<scalar_t, N, 1> &b) {
  return dot(b, a);
}

/** Cross product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::SVector<scalar_t, 3> &a,
    const ROOT::Math::SVector<scalar_t, 3> &b) {

  return ROOT::Math::Cross(a, b);
}

/** Cross product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t, class A>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::SVector<scalar_t, 3> &a,
    const ROOT::Math::VecExpr<A, scalar_t, 3> &b) {

  return ROOT::Math::Cross(a, b);
}

/** Cross product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t, class A>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::VecExpr<A, scalar_t, 3> &a,
    const ROOT::Math::SVector<scalar_t, 3> &b) {

  return ROOT::Math::Cross(a, b);
}

/** Cross product between two input vectors
 *
 * @param a the first input vector
 * @param b the second input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t, class A>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::VecExpr<A, scalar_t, 3> &a,
    const ROOT::Math::VecExpr<A, scalar_t, 3> &b) {

  return ROOT::Math::Cross(a, b);
}

/** Cross product between vector3 and matrix<3,1>
 *
 * @param a the first input vector
 * @param b the second input matrix<3,1>
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::SVector<scalar_t, 3> &a,
    const ROOT::Math::SMatrix<scalar_t, 3, 1> &b) {

  return ROOT::Math::Cross(a, b.Col(0));
}

/** Cross product between matrix<3,1> and vector3
 *
 * @param a the second input matrix<3,1>
 * @param b the first input vector
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::SMatrix<scalar_t, 3, 1> &a,
    const ROOT::Math::SVector<scalar_t, 3> &b) {

  return ROOT::Math::Cross(a.Col(0), b);
}

/** Cross product between two matrix<3,1>
 *
 * @param a the second input matrix<3,1>
 * @param b the first input matrix<3,1>
 *
 * @return a vector (expression) representing the cross product
 **/
template <typename scalar_t>
ALGEBRA_HOST inline ROOT::Math::SVector<scalar_t, 3> cross(
    const ROOT::Math::SMatrix<scalar_t, 3, 1> &a,
    const ROOT::Math::SMatrix<scalar_t, 3, 1> &b) {

  return ROOT::Math::Cross(a.Col(0), b.Col(0));
}

}  // namespace algebra::smatrix::math
