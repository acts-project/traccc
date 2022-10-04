/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/edm/container.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::device {

using prefix_sum_size_t = vecmem::data::vector_view<int>::size_type;

/// Function filling the prefix sum for a "size vector"
///
/// This simple function is just meant to translate the received "size vector"
/// into a "prefix sum". I.e. into a list of index pairs that would allow
/// visiting all elements of the jagged vector described by this "size vector".
///
/// @param[in] globalIndex The index of the current thread
/// @param[in] sizes_view View on the sizes of the "inner vectors" of a jagged
/// vector
/// @param[out] ps_view   View on the result vector of index pairs
///
TRACCC_HOST_DEVICE
void fill_prefix_sum(
    std::size_t globalIndex,
    const vecmem::data::vector_view<const prefix_sum_size_t>& sizes_view,
    vecmem::data::vector_view<prefix_sum_element_t> ps_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/device/impl/fill_prefix_sum.ipp"