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
template <std::semiregular P, typename T, typename S>
requires std::regular_invocable<P, T> __global__ void compress_adjacent(
    P projection, vecmem::data::vector_view<T> _in, S* out,
    uint32_t* out_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    vecmem::device_vector<T> in(_in);

    if (tid > 0 && tid < in.size()) {
        std::invoke_result_t<P, T> v1 = projection(in.at(tid - 1));
        std::invoke_result_t<P, T> v2 = projection(in.at(tid));

        if (v1 != v2) {
            out[atomicAdd(out_size, 1u)] = v2;
        }
    } else if (tid == 0) {
        out[atomicAdd(out_size, 1u)] = projection(in.at(tid));
    }
}

template <std::equality_comparable T>
__global__ void all_unique(const T* in, const size_t n, bool* out) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (tid_x < n && tid_y < n && tid_x != tid_y && in[tid_x] == in[tid_y]) {
        *out = false;
    }
}
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
    stream& stream, vecmem::data::vector_view<T> vector) {
    // This should never be a performance-critical step, so we can keep the
    // block size fixed.
    constexpr int block_size = 512;
    constexpr int block_size_2d = 32;

    cudaStream_t cuda_stream =
        reinterpret_cast<cudaStream_t>(stream.cudaStream());

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

    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(iout_size.get(), &initial_iout_size, sizeof(uint32_t),
                        cudaMemcpyHostToDevice, cuda_stream));
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(out.get(), &initial_out, sizeof(bool),
                        cudaMemcpyHostToDevice, cuda_stream));

    // Launch the first kernel, which will squash consecutive equal elements
    // into one element.
    kernels::compress_adjacent<P, T, projection_t>
        <<<(n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
            projection, vector, iout.get(), iout_size.get());

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy the total number of squashed elements, e.g. the size of the
    // resulting vector.
    uint32_t host_iout_size;

    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(&host_iout_size, iout_size.get(), sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, cuda_stream));

    // Launch the second kernel, which will check if the values are unique.
    uint32_t grid_size_rd =
        (host_iout_size + block_size_2d - 1) / block_size_2d;
    dim3 all_unique_grid_size(grid_size_rd, grid_size_rd);
    dim3 all_unique_block_size(block_size_2d, block_size_2d);

    kernels::all_unique<<<all_unique_grid_size, all_unique_block_size, 0,
                          cuda_stream>>>(iout.get(), host_iout_size, out.get());

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Get the result from the device and return it.
    bool host_out;

    TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(&host_out, out.get(), sizeof(bool),
                                            cudaMemcpyDeviceToHost,
                                            cuda_stream));

    stream.synchronize();

    return host_out;
}
}  // namespace traccc::cuda
