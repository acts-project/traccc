/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/kokkos/utils/definitions.hpp"

namespace traccc {
namespace kokkos {

// Some useful helper functions for kokkos device
struct kokkos_helper {

    /// reduce sum function to obtain the sum of elements in array
    ///
    /// @param array the input array
    template <typename T>
    static void reduce_sum(T* array, unsigned int block_dim, unsigned int thread_id) {
        const auto& tid = thread_id;
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
            for (int i = 1; i < block_dim / WARP_SIZE; i++) {
                array[tid] += array[i * WARP_SIZE];
            }
        }
    }

    /// Get header and item index of jagged vector
    /// This can be used when there is no null value (or invalid value) in
    /// jagged vector
    ///
    /// @param jag_vec the item jagged vector of edm
    /// @param header_idx output for header index
    /// @param item_idx output for item index
    template <typename T>
    static void find_idx_on_jagged_vector(
        const vecmem::jagged_device_vector<T>& jag_vec,
        unsigned int block_dim, 
        unsigned int block_id, 
        unsigned int thread_id,
        unsigned int& header_idx, 
        unsigned int& item_idx) {

        unsigned int ref_block_idx = 0;

        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_acc = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < jag_vec.size(); ++i) {
            nblocks_per_header = jag_vec[i].size() / block_dim + 1;
            nblocks_acc += nblocks_per_header;

            if (block_id < nblocks_acc) {
                header_idx = i;

                break;
            }
            ref_block_idx += nblocks_per_header;
        }
        item_idx = (block_id - ref_block_idx) * block_dim + thread_id;
    }

    /// Get header and item index of edm container
    /// This can be used when there are null values in jagged vector and the
    /// header indicates the number of effective elements
    ///
    /// @param jag_vec the item jagged vector of edm
    /// @param header_idx output for header index
    /// @param item_idx output for item index
    template <typename header_t, typename item_t>
    static void find_idx_on_container(
        const device_container<header_t, item_t>& container,
        unsigned int block_dim, 
        unsigned int block_id, 
        unsigned int thread_id,
        unsigned int& header_idx, 
        unsigned int& item_idx) {

        unsigned int ref_block_idx = 0;

        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < container.size(); ++i) {
            nblocks_per_header =
                container.get_headers()[i].get_ref_num() / block_dim + 1;
            nblocks_accum += nblocks_per_header;

            if (block_id < nblocks_accum) {
                header_idx = i;

                break;
            }

            ref_block_idx += nblocks_per_header;
        }
        item_idx = (block_id - ref_block_idx) * block_dim + thread_id;
    }
};

}  // namespace kokkos
}  // namespace traccc
