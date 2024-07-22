/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include <traccc/sycl/utils/queue_wrapper.hpp>

#include "../utils/get_queue.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

// SYCL include
#include <CL/sycl.hpp>

// System include
#include <concepts>

namespace traccc::sycl {
namespace kernels {
template <typename R, typename T>
class IsOrderedOn {};
}  // namespace kernels

/**
 * @brief Sanity check that a given vector is ordered on a given relation.
 *
 * For a vector $v$ to be ordered on a relation $R$, it must be the case that
 * for all indices $i$ and $j$, if $i < j$, then $R(i, j)$.
 *
 * @note This function runs in O(n) time.
 *
 * @note Although functions like `std::sort` requires the relation to be strict
 * weak order, this function is more lax in its requirements. Rather, the
 * relation should be a total preorder, i.e. a non-strict weak order.
 *
 * @note For any strict weak order $R$, `is_ordered_on(sort(R, v))` is true.
 *
 * @tparam R The type of relation $R$, a callable which returns a bool if the
 * first argument can be immediately before the second type.
 * @tparam T The type of the vector.
 * @param relation A relation object of type `R`.
 * @param mr A memory resource used for allocating intermediate memory.
 * @param vector The vector which to check for ordering.
 * @return true If the vector is ordered on `R`.
 * @return false Otherwise.
 */
template <std::semiregular R, typename T>
requires std::relation<R, T, T> bool is_ordered_on(
    R relation, vecmem::memory_resource& mr, vecmem::copy& copy,
    queue_wrapper& queue_wrapper, vecmem::data::vector_view<T> vector) {
    // This should never be a performance-critical step, so we can keep the
    // block size fixed.
    constexpr int block_size = 512;

    cl::sycl::queue& queue = details::get_queue(queue_wrapper);

    // Grab the number of elements in our vector.
    uint32_t n = copy.get_size(vector);

    // Initialize the output boolean.
    vecmem::unique_alloc_ptr<bool> out = vecmem::make_unique_alloc<bool>(mr);
    bool initial_out = true;

    cl::sycl::event kernel1_memcpy1 =
        queue.memcpy(out.get(), &initial_out, sizeof(bool));

    cl::sycl::nd_range<1> kernel_range{
        cl::sycl::range<1>(((n + block_size - 1) / block_size) * block_size),
        cl::sycl::range<1>(block_size)};

    cl::sycl::event kernel1 = queue.submit([&](cl::sycl::handler& h) {
        h.depends_on(kernel1_memcpy1);
        h.parallel_for<kernels::IsOrderedOn<R, T>>(
            kernel_range, [=, out = out.get()](cl::sycl::nd_item<1> item) {
                std::size_t tid = item.get_global_linear_id();

                vecmem::device_vector<T> in(vector);

                if (tid > 0 && tid < in.size()) {
                    if (!relation(in.at(tid - 1), in.at(tid))) {
                        *out = false;
                    }
                }
            });
    });

    // Copy the output to host, then return it.
    bool host_out;

    queue.memcpy(&host_out, out.get(), sizeof(bool), {kernel1})
        .wait_and_throw();

    return host_out;
}
}  // namespace traccc::sycl
