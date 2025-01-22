/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <variant>

namespace traccc::device {

/** @struct make_result_t
 * @brief Helper class for return type of make_sum_buffer function
 * @param view View on the result vector
 * @param variant Object posessing memory of the result vector
 * @param totalSize Last member of result vector
 */
struct prefix_sum_buffer {
    vecmem::data::vector_view<prefix_sum_size_t> view;
    std::variant<vecmem::vector<prefix_sum_size_t>,
                 vecmem::data::vector_buffer<prefix_sum_size_t>>
        variant;
    prefix_sum_size_t totalSize;
};  // struct prefix_sum_buffer

using prefix_sum_buffer_t = prefix_sum_buffer;

/// Function providing the prefix sum for a "size vector"
///
/// This simple function is just meant to translate the received "size vector"
/// into its prefix sum. I.e. Each element of the output vector equals the
/// summation of all the elements up to that point of the input vector
///
/// @param sizes The sizes of the "inner vectors" of a jagged vector
/// @param copy A "copy object" capable of dealing with the view
/// @param mr    The memory resource to use of the result
/// @return A class containing the prefix sum view, a memory posessing object on
/// the view elements, and the total summation of the size vector
///
TRACCC_HOST
inline prefix_sum_buffer_t make_prefix_sum_buffer(
    const std::vector<prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr) {

    if (sizes.size() == 0) {
        return prefix_sum_buffer_t{
            vecmem::data::vector_view<prefix_sum_size_t>{},
            std::variant<vecmem::vector<prefix_sum_size_t>,
                         vecmem::data::vector_buffer<prefix_sum_size_t>>{},
            prefix_sum_size_t{0}};
    }

    // Create vector with summation of sizes
    vecmem::vector<prefix_sum_size_t> sizes_sum(sizes.size(),
                                                mr.host ? mr.host : &(mr.main));
    std::partial_sum(sizes.begin(), sizes.end(), sizes_sum.begin(),
                     std::plus<prefix_sum_size_t>());
    const prefix_sum_size_t totalSize = sizes_sum.back();

    if (mr.host != nullptr) {
        // Create buffer and view objects
        vecmem::data::vector_buffer<prefix_sum_size_t> sizes_sum_buff(
            static_cast<unsigned int>(sizes_sum.size()), mr.main);
        copy.setup(sizes_sum_buff)->ignore();
        (copy)(vecmem::get_data(sizes_sum), sizes_sum_buff)->wait();
        vecmem::data::vector_view<prefix_sum_size_t> sizes_sum_view(
            sizes_sum_buff);

        return {sizes_sum_view, std::move(sizes_sum_buff), totalSize};
    } else {
        // Create view object
        vecmem::data::vector_view<prefix_sum_size_t> sizes_sum_view =
            vecmem::get_data(sizes_sum);

        return {sizes_sum_view, std::move(sizes_sum), totalSize};
    }
}
}  // namespace traccc::device
