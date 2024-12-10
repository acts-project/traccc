/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <algorithm>
#include <iterator>
#include <type_traits>

namespace vecmem {
namespace details {

/// Helper trait for identifying input iterators
///
/// It comes in handy in some of the functions of the custom (device)
/// container types that use templated iterator values. Which could hide
/// overloads of the same function with the same number of (non-templated)
/// arguments.
///
/// The implementation is *very* simplistic at the moment. It could/should
/// be made more elaborate when the need arises.
///
template <typename iterator_type, typename value_type>
using is_iterator_of = std::is_convertible<
    typename std::iterator_traits<iterator_type>::value_type, value_type>;

/// Helper trait for detecting when a type is a non-const version of another
///
/// This comes into play multiple times to enable certain constructors
/// conditionally through SFINAE.
///
template <typename CTYPE, typename NCTYPE>
struct is_same_nc {
    static constexpr bool value = false;
};

template <typename TYPE>
struct is_same_nc<const TYPE, TYPE> {
    static constexpr bool value = true;
};

/// Implementation for @c std::conjunction
///
/// Since @c std::conjunction is only available starting with C++17, but it
/// comes in very handy in some places in the VecMem code, this is a naive
/// custom implementation for it.
///
template <class...>
struct conjunction : std::true_type {};

template <class B1>
struct conjunction<B1> : B1 {};

template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <class... B>
constexpr bool conjunction_v = conjunction<B...>::value;

/// Implementation for @c std::disjunction
///
/// Since @c std::disjunction is only available starting with C++17, but it
/// comes in very handy in some places in the VecMem code, this is a naive
/// custom implementation for it.
///
template <class...>
struct disjunction : std::false_type {};

template <class B1>
struct disjunction<B1> : B1 {};

template <class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>> {};

template <class... B>
constexpr bool disjunction_v = disjunction<B...>::value;

/// Implementation for @c std::negation
///
/// Since @c std::negation is only available starting with C++17, but it
/// comes in very handy in some places in the VecMem code, this is a naive
/// custom implementation for it.
///
template <class B>
struct negation : std::integral_constant<bool, !bool(B::value)> {};

template <class B>
constexpr bool negation_v = negation<B>::value;

/// Find the maximum of a variadic number of elements, terminal function
/// @tparam T The type of the (final) element
/// @param t The value of the (final) element
/// @return  The value of the (final) element
template <typename T>
auto max(T&& t) {
    return std::forward<T>(t);
}

/// Find the maximum of a variadic number of elements, recursive function
/// @tparam T  The type of the (next) element
/// @tparam Ts The types of the remaining element(s)
/// @param t   The value of the (next) element
/// @param ts  The values of the remaining element(s)
/// @return    The maximum of the elements
template <typename T, typename... Ts>
auto max(T&& t, Ts&&... ts) {
    return std::max(std::forward<T>(t), max(std::forward<Ts>(ts)...));
}

/// Type trait that indicates whether a given type is an implicit lifetime
/// type.
///
/// @note The definition of "implicit lifetime type" differs a lot across C++
/// standards, as does the implementation of this type trait. In C++17 and
/// earlier, the concept of such types did not exist. In C++20, these types
/// were defined but no type trait for them was available. In C++23, a type
/// trait is available.
///
/// @warning On pre-C++17 translation units, this type trait is always assumed
/// to be true.
///
/// @{
#if defined(__cpp_lib_is_implicit_lifetime) && \
    __cpp_lib_is_implicit_lifetime >= 202302L
template <class TYPE>
using is_implicit_lifetime = std::is_implicit_lifetime<TYPE>;
#define VECMEM_HAVE_IS_IMPLICIT_LIFETIME
#elif defined(__cpp_lib_is_aggregate) && __cpp_lib_is_aggregate >= 201703L
// Implementation taken directly from P2674R1.
template <class TYPE>
struct is_implicit_lifetime
    : disjunction<
          std::is_scalar<TYPE>, std::is_array<TYPE>, std::is_aggregate<TYPE>,
          conjunction<
              std::is_trivially_destructible<TYPE>,
              disjunction<std::is_trivially_default_constructible<TYPE>,
                          std::is_trivially_copy_constructible<TYPE>,
                          std::is_trivially_move_constructible<TYPE>>>> {};
#define VECMEM_HAVE_IS_IMPLICIT_LIFETIME
#else
// If we are on such an old version of C++, we're basically in the wild west,
// so we allow the user to do whatever they want.
template <class TYPE>
using is_implicit_lifetime = std::true_type;
#endif

template <class TYPE>
constexpr bool is_implicit_lifetime_v = is_implicit_lifetime<TYPE>::value;
/// @}
}  // namespace details
}  // namespace vecmem
