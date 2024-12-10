/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstddef>

namespace vecmem {

/// @brief Namespace for types that should not be used directly by clients
///
/// These would be classes/functions/types that are needed by the public
/// interface of classes/functions outside of this namespace.
///
namespace details {

/// Helper type for an array in a static_vector with a given type and size
///
/// This is needed to handle zero-sized arrays correctly. As those are not
/// part of the C++ standard.
///
template <typename T, std::size_t size>
struct static_vector_type {
    typedef T type[size];
};

template <typename T>
struct static_vector_type<T, 0> {
    typedef T* type;
};

}  // namespace details
}  // namespace vecmem
