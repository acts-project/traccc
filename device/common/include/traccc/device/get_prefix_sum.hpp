/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Thrust include(s).
#include <thrust/pair.h>

// System include(s).
#include <utility>

namespace traccc::device {

/// Type for the individual elements in the prefix sum vector
typedef thrust::pair<std::size_t, std::size_t> prefix_sum_element_t;

/// Convenience type definition for the return value of the helper function
typedef vecmem::vector<prefix_sum_element_t> prefix_sum_t;

/// Get the prefix sum for a generic jagged vector view
///
/// @param view The jagged vector view to make the prefix sum for
/// @param mr   The memory resource to use for the result
/// @param copy A "copy object" capable of dealing with the view
/// @return The prefix sum for efficiently iterating over all elements of the
///         jagged vector
///
template <typename element_t>
TRACCC_HOST prefix_sum_t
get_prefix_sum(const vecmem::data::jagged_vector_view<element_t>& view,
               vecmem::memory_resource& mr, vecmem::copy& copy);

/// Get the prefix sum for a generic jagged vector buffer
///
/// @param buffer The jagged vector buffer to make the prefix sum for
/// @param mr     The memory resource to use for the result
/// @param copy   A "copy object" capable of dealing with the buffer
/// @return The prefix sum for efficiently iterating over all elements of the
///         jagged vector
///
template <typename element_t>
TRACCC_HOST prefix_sum_t
get_prefix_sum(const vecmem::data::jagged_vector_buffer<element_t>& buffer,
               vecmem::memory_resource& mr, vecmem::copy& copy);

/// Function providing the prefix sum for a "size vector"
///
/// This simple function is just meant to translate the received "size vector"
/// into a "prefix sum". I.e. into a list of index pairs that would allow
/// visiting all elements of the jagged vector described by this "size vector".
///
/// @param sizes The sizes of the "inner vectors" of a jagged vector
/// @param mr    The memory resource to use of the result
/// @return The prefix sum for efficiently iterating over all element of a
///         jagged vector
///
TRACCC_HOST
prefix_sum_t get_prefix_sum(
    const std::vector<vecmem::data::vector_view<int>::size_type>& sizes,
    vecmem::memory_resource& mr);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/device/impl/get_prefix_sum.ipp"
