/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

/// Simple struct holding the types that describe a given plugin
template <
    typename scalar_t, typename point2_t, typename point3_t, typename vector2_t,
    typename vector3_t, typename transform3_t, typename size_ty,
    template <typename, size_ty, size_ty> class matrix_t, class matrix_actor_t>
struct test_types {

  using scalar = scalar_t;
  using point2 = point2_t;
  using point3 = point3_t;
  using vector2 = vector2_t;
  using vector3 = vector3_t;
  using transform3 = transform3_t;
  using size_type = size_ty;
  template <size_type ROWS, size_type COLS>
  using matrix = matrix_t<scalar, ROWS, COLS>;
  using matrix_actor = matrix_actor_t;

};  // struct test_types
