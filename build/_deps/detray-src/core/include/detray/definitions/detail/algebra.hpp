/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if DETRAY_ALGEBRA_ARRAY
#include "detray/plugins/algebra/array_definitions.hpp"
#elif DETRAY_ALGEBRA_EIGEN
#include "detray/plugins/algebra/eigen_definitions.hpp"
#elif DETRAY_ALGEBRA_SMATRIX
#include "detray/plugins/algebra/smatrix_definitions.hpp"
#elif DETRAY_ALGEBRA_VC_AOS
#include "detray/plugins/algebra/vc_aos_definitions.hpp"
#elif DETRAY_ALGEBRA_VC_SOA
#include "detray/plugins/algebra/vc_soa_definitions.hpp"
#else
#error "No algebra plugin selected! Please link to one of the algebra plugins."
#endif

// Project include(s)
#include "detray/utils/concepts.hpp"

// System include(s)
#include <type_traits>

namespace detray {

namespace detail {
/// The detray scalar types (can be SIMD)
/// @{
template <typename T>
struct get_scalar {};

// TODO replace by scalar concept from algebra-plugins
template <concepts::arithmetic T>
struct get_scalar<T> {
    using scalar = T;
};

template <typename T>
requires(!std::same_as<typename T::scalar, void>) struct get_scalar<T> {
    using scalar = typename T::scalar;
};
/// @}

/// The detray algebra types (can be SIMD)
/// @{
template <typename T>
struct get_algebra {};

template <typename T>
requires(!std::same_as<typename T::point3D, void>) struct get_algebra<T> {
    using point2D = typename T::point2D;
    using point3D = typename T::point3D;
    using vector3D = typename T::vector3D;
    using transform3D = typename T::transform3D;
};
/// @}

/// The detray matrix types
/// @{
template <typename T>
struct get_matrix {};

template <typename T>
requires(
    !std::same_as<typename T::matrix_operator, void>) struct get_matrix<T> {
    using matrix_operator = typename T::matrix_operator;
    using size_type = typename matrix_operator::size_ty;

    template <std::size_t ROWS, std::size_t COLS>
    using matrix = typename matrix_operator::template matrix_type<
        static_cast<size_type>(ROWS), static_cast<size_type>(COLS)>;
};
/// @}

}  // namespace detail

template <template <typename> class A, typename T>
using dsimd = typename A<float>::template simd<T>;

template <typename A = detray::scalar>
using dscalar = typename detail::get_scalar<A>::scalar;

template <typename A>
using dpoint2D = typename detail::get_algebra<A>::point2D;

template <typename A>
using dpoint3D = typename detail::get_algebra<A>::point3D;

template <typename A>
using dvector3D = typename detail::get_algebra<A>::vector3D;

template <typename A>
using dtransform3D = typename detail::get_algebra<A>::transform3D;

template <typename A>
using dmatrix_operator = typename detail::get_matrix<A>::matrix_operator;

template <typename A>
using dsize_type = typename detail::get_matrix<A>::size_type;

template <typename A, std::size_t R, std::size_t C>
using dmatrix = typename detail::get_matrix<A>::template matrix<R, C>;

namespace concepts {

/// Check if an algebra has soa layout
/// @{
template <typename A>
concept soa_algebra = (!concepts::arithmetic<dscalar<A>>);

template <typename A>
concept aos_algebra = (!concepts::soa_algebra<A>);
/// @}

}  // namespace concepts

}  // namespace detray
