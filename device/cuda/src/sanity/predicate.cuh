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
template <typename P, typename T>
requires std::predicate<P, T> __global__ void true_for_all_kernel(
    P projection, vecmem::data::vector_view<T> _in, bool* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    vecmem::device_vector<T> in(_in);

    if (tid < in.size()) {
        if (!projection(in.at(tid))) {
            *out = false;
        }
    }
}
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
    stream& stream, vecmem::data::vector_view<T> vector) {
    // This should never be a performance-critical step, so we can keep the
    // block size fixed.
    constexpr int block_size = 512;

    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(stream.cudaStream());

    // Grab the number of elements in our vector.
    const std::uint32_t n = copy.get_size(vector);

    // Allocate memory for outputs, then set them up.
    vecmem::unique_alloc_ptr<bool> device_out =
        vecmem::make_unique_alloc<bool>(mr);

    bool initial_out = true;

    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(device_out.get(), &initial_out, sizeof(bool),
                        cudaMemcpyHostToDevice, cuda_stream));

    // Launch the main kernel.
    kernels::true_for_all_kernel<P, T>
        <<<(n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
            predicate, vector, device_out.get());

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy the total number of squashed elements, e.g. the size of the
    // resulting vector.
    bool host_out;

    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(&host_out, device_out.get(), sizeof(bool),
                        cudaMemcpyDeviceToHost, cuda_stream));

    stream.synchronize();

    return host_out;
}

template <typename P, typename T>
requires std::predicate<P, T> bool false_for_all(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    stream& stream, vecmem::data::vector_view<T> vector) {
    return true_for_all(
        [projection] __device__<typename... Args>(Args && ... args) {
            return !projection(std::forward<Args>(args)...);
        },
        mr, copy, stream, vector);
}

template <typename P, typename T>
requires std::predicate<P, T> bool true_for_any(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    stream& stream, vecmem::data::vector_view<T> vector) {
    return !false_for_all(std::forward<P>(projection), mr, copy, stream,
                          vector);
}

template <typename P, typename T>
requires std::predicate<P, T> bool false_for_any(
    P&& projection, vecmem::memory_resource& mr, vecmem::copy& copy,
    stream& stream, vecmem::data::vector_view<T> vector) {
    return !true_for_all(std::forward<P>(projection), mr, copy, stream, vector);
}
}  // namespace traccc::cuda
