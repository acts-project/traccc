/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

// Some useful helper functions for cuda device
struct cuda_helper {

    /// reduce sum function to obtain the sum of elements in array
    ///
    /// @param array the input array    
    template <typename T>
    static __device__ void reduce_sum(T* array) {
	const auto& tid = threadIdx.x;
        array[tid] +=
            __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 2, WARP_SIZE);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 4,
                                       WARP_SIZE / 2);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 8,
                                       WARP_SIZE / 4);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 16,
                                       WARP_SIZE / 8);
        array[tid] += __shfl_down_sync(0xFFFFFFFF, array[tid], WARP_SIZE / 32,
                                       WARP_SIZE / 16);

        __syncthreads();

        if (tid == 0) {
            for (int i = 1; i < blockDim.x / WARP_SIZE; i++) {
                array[tid] += array[i * WARP_SIZE];
            }
        }
    }

    /// Get index of header vector of event data container for a given block ID.
    ///
    /// @param jag_vec the item jagged vector of edm 
    /// @param header_idx the header idx
    /// @param ref_block_idx the reference block idx for a given header idx    
    template <typename T>
    static __device__ void get_header_idx(
        const vecmem::jagged_device_vector<T>& jag_vec, unsigned int& header_idx,
        unsigned int& ref_block_idx) {

	/// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

	/// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < jag_vec.size(); ++i) {
            nblocks_per_header = jag_vec[i].size() / blockDim.x + 1;
            nblocks_accum += nblocks_per_header;

            if (blockIdx.x < nblocks_accum) {
                header_idx = i;

                break;
            }

            ref_block_idx += nblocks_per_header;
        }
    }

    /// Get index of header vector of event data container for a given block ID.
    ///
    /// @param container event data container where header element indicates the number of elements in item vector
    /// @param header_idx the header idx
    /// @param ref_block_idx the reference block idx for a given header idx        
    template <typename header_t, typename item_t>
    static __device__ void get_header_idx(
        const device_container<header_t, item_t>& container,
        unsigned int& header_idx, unsigned int& ref_block_idx) {

	/// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

	/// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < container.headers.size(); ++i) {
            nblocks_per_header = container.headers[i] / blockDim.x + 1;
            nblocks_accum += nblocks_per_header;

            if (blockIdx.x < nblocks_accum) {
                header_idx = i;

                break;
            }

            ref_block_idx += nblocks_per_header;
        }
    }
};

}  // namespace cuda
}  // namespace traccc
