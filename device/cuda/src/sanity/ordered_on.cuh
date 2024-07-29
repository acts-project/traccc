/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "traccc/cuda/utils/stream.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

// CUDA include
#include <cuda_runtime.h>

// System include
#include <concepts>

namespace traccc::cuda {
namespace kernels {
template <std::semiregular R, typename T>
requires std::relation<R, T, T> __global__ void is_ordered_on_kernel(
    R relation, vecmem::data::vector_view<T> _in, bool* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    vecmem::device_vector<T> in(_in);

    if (tid > 0 && tid < in.size()) {
        if (!relation(in.at(tid - 1), in.at(tid))) {
            *out = false;
        }
    }
}
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
    R relation, vecmem::memory_resource& mr, vecmem::copy& copy, stream& stream,
    vecmem::data::vector_view<T> vector) {
    // This should never be a performance-critical step, so we can keep the
    // block size fixed.
    constexpr int block_size = 512;

    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(stream.cudaStream());

    // Grab the number of elements in our vector.
    uint32_t n = copy.get_size(vector);

    // Initialize the output boolean.
    vecmem::unique_alloc_ptr<bool> out = vecmem::make_unique_alloc<bool>(mr);
    bool initial_out = true;
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(out.get(), &initial_out, sizeof(bool),
                        cudaMemcpyHostToDevice, cuda_stream));

    // Launch the kernel which will write its result to the `out` boolean.
    kernels::is_ordered_on_kernel<<<(n + block_size - 1) / block_size,
                                    block_size, 0, cuda_stream>>>(
        relation, vector, out.get());

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy the output to host, then return it.
    bool host_out;

    TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_out, out.get(), sizeof(bool),
                                            cudaMemcpyDeviceToHost,
                                            cuda_stream));

    stream.synchronize();

    return host_out;
}
}  // namespace traccc::cuda
