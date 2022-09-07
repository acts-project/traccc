/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// System include(s).
#include <type_traits>

namespace traccc {

/// Trivial class mimicking @c std::pair
///
/// The purpose of this class is to make an @c std::pair type that is fully
/// functional in "device code". Unfortunately some parts of @c std::pair are
/// not at the moment of writing.
///
/// @tparam T1 The type of the first element of the pair
/// @tparam T2 The type of the second element of the pair
///
template <typename T1, typename T2>
struct pair {

    /// The type of the first element
    using first_type = T1;
    /// The type of the second element
    using second_type = T2;

    /// Default constructor
    TRACCC_HOST_DEVICE
    pair();

    /// Constructor copying existing elements
    TRACCC_HOST_DEVICE
    pair(const first_type& f, const second_type& s);
    /// Constructor moving existing elements
    TRACCC_HOST_DEVICE
    pair(first_type&& f, second_type&& s);

    /// Copy constructor
    TRACCC_HOST_DEVICE
    pair(const pair& parent);
    /// Move constructor
    TRACCC_HOST_DEVICE
    pair(pair&& parent);

    /// Copy constructor from a different type
    template <typename U1, typename U2,
              std::enable_if_t<std::is_convertible<U1, T1>::value &&
                                   std::is_convertible<U2, T2>::value,
                               bool> = true>
    TRACCC_HOST_DEVICE pair(const pair<U1, U2>& parent);
    /// Move constructor from a different type
    template <typename U1, typename U2,
              std::enable_if_t<std::is_convertible<U1, T1>::value &&
                                   std::is_convertible<U2, T2>::value,
                               bool> = true>
    TRACCC_HOST_DEVICE pair(pair<U1, U2>&& parent);

    /// Copy assignment
    TRACCC_HOST_DEVICE
    pair& operator=(const pair& rhs);
    /// Move assignment
    TRACCC_HOST_DEVICE
    pair& operator=(pair&& rhs);

    /// The first element
    first_type first;
    /// The second element
    second_type second;

};  // struct pair

}  // namespace traccc

// Include the implementation.
#include "traccc/utils/impl/pair.ipp"
