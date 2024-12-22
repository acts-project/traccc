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
template <typename P, typename T>
class TrueForAllPredicate {};
}  // namespace kernels

/**
 * @brief Sanity check that a predicate is true for all elements of a vector.
 *
 * @note This function runs in O(n) time.
 *
 * @tparam P The type of the predicate.
 * @tparam T The type of the vector.
 * @param predicate A projection object of type `P`.
 * @param mr A memory resource used for allocating intermediate memory.
 * @param copy A copy object.
 * @param stream A wrapped CUDA stream.
 * @param vector The vector which to check for contiguity.
 * @return true If `predicate` is true for all elements of `vector`.
 * @return false Otherwise.
 */
template <typename P, typename T>
requires std::predicate<P, T> bool true_for_all(
    P&& predicate, vecmem::memory_resource& mr, vecmem::copy& copy,
    queue_wrapper& queue_wrapper, vecmem::data::vector_view<T> vector) {
    // This should never be a performance-critical step, so we can keep the
    // block size fixed.
    constexpr int block_size = 512;

    cl::sycl::queue& queue = details::get_queue(queue_wrapper);

    // Grab the number of elements in our vector.
    const std::uint32_t n = copy.get_size(vector);

    // Allocate memory for outputs, then set them up.
    vecmem::unique_alloc_ptr<bool> device_out =
        vecmem::make_unique_alloc<bool>(mr);

    bool initial_out = true;

    cl::sycl::event kernel1_memcpy1 =
        queue.memcpy(device_out.get(), &initial_out, sizeof(bool));

    // Launch the main kernel.
    cl::sycl::nd_range<1> kernel_range{
        cl::sycl::range<1>(((n + block_size - 1) / block_size) * block_size),
        cl::sycl::range<1>(block_size)};

    cl::sycl::event kernel1 = queue.submit([&](cl::sycl::handler& h) {
        h.depends_on(kernel1_memcpy1);
        h.parallel_for<kernels::TrueForAllPredicate<P, T>>(
            kernel_range, [projection, vector,
                           out = device_out.get()](cl::sycl::nd_item<1> item) {
                std::size_t tid = item.get_global_linear_id();

                vecmem::device_vector<T> in(vector);

                if (tid < in.size()) {
                    if (!projection(in.at(tid))) {
                        *out = false;
                    }
                }
            });
    });

    // Copy the total number of squashed elements, e.g. the size of the
    // resulting vector.
    bool host_out;

    queue.memcpy(&host_out, out.get(), sizeof(bool), {kernel1})
        .wait_and_throw();

    return host_out;
}

template <typename P, typename T>
requires std::predicate<P, T> bool false_for_all(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    queue_wrapper& queue_wrapper, vecmem::data::vector_view<T> vector) {
    return true_for_all(
        [projection]<typename... Args>(Args && ... args) {
            return !projection(std::forward<Args>(args)...);
        },
        mr, copy, queue_wrapper, vector);
}

template <typename P, typename T>
requires std::predicate<P, T> bool true_for_any(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    queue_wrapper& queue_wrapper, vecmem::data::vector_view<T> vector) {
    return !false_for_all(std::forward<P>(projection), mr, copy, queue_wrapper,
                          vector);
}

template <typename P, typename T>
requires std::predicate<P, T> bool false_for_any(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    queue_wrapper& queue_wrapper, vecmem::data::vector_view<T> vector) {
    return !true_for_all(std::forward<P>(projection), mr, copy, queue_wrapper,
                         vector);
}
}  // namespace traccc::sycl
