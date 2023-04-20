/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include "vecmem/utils/cuda/copy.hpp"

namespace traccc::cuda {

/// Function that returns vector of prefix_sum_element_t for accessing a jagged
/// vector's elements in device Example: Jagged vector with sizes = {3,2,1,...}
/// Returns: {[0,0], [0,1], [0,2], [1,0], [1,1], [2,0], ...}
///
/// @param[in] sizes       The sizes of the jagged vector
/// @param copy A "copy object" capable of dealing with the view
/// @param mr   The memory resource to create the buffer with
/// @return     A vector buffer of prefix_sum element
///
vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr);

/// Function that returns vector of prefix_sum_element_t for accessing a jagged
/// vector's elements in device Example: Jagged vector with sizes = {3,2,1,...}
/// Returns: {[0,0], [0,1], [0,2], [1,0], [1,1], [2,0], ...}
///
/// @param[in] sizes       The sizes of the jagged vector
/// @param copy A "copy object" capable of dealing with the view
/// @param mr   The memory resource to create the buffer with
/// @param str  The CUDA stream to create the buffer with
/// @return     A vector buffer of prefix_sum element
///
vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr, const stream& str);

}  // namespace traccc::cuda
