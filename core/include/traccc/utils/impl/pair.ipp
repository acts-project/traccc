/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>::pair() : first{}, second{} {}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>::pair(const first_type& f, const second_type& s)
    : first(f), second(s) {}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>::pair(first_type&& f, second_type&& s)
    : first(f), second(s) {}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>::pair(const pair& parent)
    : first(parent.first), second(parent.second) {}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>::pair(pair&& parent)
    : first(parent.first), second(parent.second) {}

template <typename T1, typename T2>
template <typename U1, typename U2,
          std::enable_if_t<std::is_convertible<U1, T1>::value &&
                               std::is_convertible<U2, T2>::value,
                           bool> >
TRACCC_HOST_DEVICE pair<T1, T2>::pair(const pair<U1, U2>& parent)
    : first(parent.first), second(parent.second) {}

template <typename T1, typename T2>
template <typename U1, typename U2,
          std::enable_if_t<std::is_convertible<U1, T1>::value &&
                               std::is_convertible<U2, T2>::value,
                           bool> >
TRACCC_HOST_DEVICE pair<T1, T2>::pair(pair<U1, U2>&& parent)
    : first(parent.first), second(parent.second) {}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>& pair<T1, T2>::operator=(const pair& rhs) {

    // Prevent self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Assign the elements.
    this->first = rhs.first;
    this->second = rhs.second;

    // Return a reference to the updated object.
    return *this;
}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE pair<T1, T2>& pair<T1, T2>::operator=(pair&& rhs) {

    // Prevent self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Assign the elements.
    this->first = rhs.first;
    this->second = rhs.second;

    // Return a reference to the updated object.
    return *this;
}

}  // namespace traccc
