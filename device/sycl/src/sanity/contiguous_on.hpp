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
#include <vecmem/memory/device_atomic_ref.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

// SYCL include
#include <CL/sycl.hpp>

// System include
#include <concepts>

namespace traccc::sycl {
namespace kernels {
template <typename P, typename T, typename S>
class IsContiguousOnCompressAdjacent {};

template <typename T>
class IsContiguousOnAllUnique {};
}  // namespace kernels

/**
 * @brief Sanity check that a given vector is contiguous on a given projection.
 *
 * For a vector $v$ to be contiguous on a projection $\pi$, it must be the case
 * that for all indices $i$ and $j$, if $v_i = v_j$, then all indices $k$
 * between $i$ and $j$, $v_i = v_j = v_k$.
 *
 * @note This function runs in O(n^2) time.
 *
 * @tparam P The type of projection $\pi$, a callable which returns some
 * comparable type.
 * @tparam T The type of the vector.
 * @param projection A projection object of type `P`.
 * @param mr A memory resource used for allocating intermediate memory.
 * @param vector The vector which to check for contiguity.
 * @return true If the vector is contiguous on `P`.
 * @return false Otherwise.
 */
template <std::semiregular P, std::equality_comparable T>
requires std::regular_invocable<P, T> bool is_contiguous_on(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    queue_wrapper& queue_wrapper, vecmem::data::vector_view<T> vector) {
    // This should never be a performance-critical step, so we can keep the
    // block size fixed.
    constexpr int block_size = 512;

    cl::sycl::queue& queue = details::get_queue(queue_wrapper);

    // Grab the number of elements in our vector.
    uint32_t n = copy.get_size(vector);

    // Get the output type of the projection.
    using projection_t = std::invoke_result_t<P, T>;

    // Allocate memory for intermediate values and outputs, then set them up.
    vecmem::unique_alloc_ptr<projection_t[]> iout =
        vecmem::make_unique_alloc<projection_t[]>(mr, n);
    vecmem::unique_alloc_ptr<uint32_t> iout_size =
        vecmem::make_unique_alloc<uint32_t>(mr);
    vecmem::unique_alloc_ptr<bool> out = vecmem::make_unique_alloc<bool>(mr);

    uint32_t initial_iout_size = 0;
    bool initial_out = true;

    cl::sycl::event kernel1_memcpy1_evt =
        queue.copy(&initial_iout_size, iout_size.get(), 1);
    cl::sycl::event kernel2_memcpy1_evt =
        queue.copy(&initial_out, out.get(), 1);

    cl::sycl::nd_range<1> compress_adjacent_range{
        cl::sycl::range<1>(((n + block_size - 1) / block_size) * block_size),
        cl::sycl::range<1>(block_size)};

    // Launch the first kernel, which will squash consecutive equal elements
    // into one element.
    cl::sycl::event kernel1_evt = queue.submit([&](cl::sycl::handler& h) {
        h.depends_on(kernel1_memcpy1_evt);
        h.parallel_for<
            kernels::IsContiguousOnCompressAdjacent<P, T, projection_t>>(
            compress_adjacent_range,
            [vector, projection, out = iout.get(),
             out_size = iout_size.get()](cl::sycl::nd_item<1> item) {
                std::size_t tid = item.get_global_linear_id();

                vecmem::device_vector<T> in(vector);
                vecmem::device_atomic_ref<uint32_t> out_siz_atm(*out_size);

                if (tid > 0 && tid < in.size()) {
                    std::invoke_result_t<P, T> v1 = projection(in.at(tid - 1));
                    std::invoke_result_t<P, T> v2 = projection(in.at(tid));

                    if (v1 != v2) {
                        out[out_siz_atm.fetch_add(1)] = v2;
                    }
                } else if (tid == 0) {
                    out[out_siz_atm.fetch_add(1)] = projection(in.at(tid));
                }
            });
    });

    // Copy the total number of squashed elements, e.g. the size of the
    // resulting vector.
    uint32_t host_iout_size;

    queue
        .memcpy(&host_iout_size, iout_size.get(), sizeof(uint32_t),
                {kernel1_evt})
        .wait_and_throw();

    uint32_t grid_size_rd = (host_iout_size + block_size - 1) / block_size;

    cl::sycl::nd_range<2> all_unique_range{
        cl::sycl::range<2>(grid_size_rd * block_size, host_iout_size),
        cl::sycl::range<2>(block_size, 1)};

    // Launch the second kernel, which will check if the values are unique.
    cl::sycl::event kernel2_evt = queue.submit([&](cl::sycl::handler& h) {
        h.depends_on(kernel2_memcpy1_evt);
        h.parallel_for<kernels::IsContiguousOnAllUnique<T>>(
            all_unique_range, [n = host_iout_size, in = iout.get(),
                               out = out.get()](cl::sycl::nd_item<2> item) {
                std::size_t tid_x = item.get_global_id(0);
                std::size_t tid_y = item.get_global_id(1);

                if (tid_x < n && tid_y < n && tid_x != tid_y &&
                    in[tid_x] == in[tid_y]) {
                    *out = false;
                }
            });
    });

    // Get the result from the device and return it.
    bool host_out;

    queue.memcpy(&host_out, out.get(), sizeof(bool), {kernel2_evt})
        .wait_and_throw();

    return host_out;
}
}  // namespace traccc::sycl
