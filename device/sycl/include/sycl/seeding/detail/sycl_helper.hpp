/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <CL/sycl.hpp>

namespace traccc {
namespace sycl {

// Short aliast for accessor to local memory (shared memory in CUDA)
template <typename T>
using local_accessor = ::sycl::accessor<
    T,
    1,
    ::sycl::access::mode::read_write,
    ::sycl::access::target::local>;

// Some useful helper funcitons
struct sycl_helper {

    /// Function that performs reduction on the local memory using subgroup operations
    ///
    /// @param array pointer to an array in local memory
    /// @param item sycl nd_item for quering local indexes and groups
    static
    void reduceInShared(::sycl::multi_ptr<int, ::sycl::access::address_space::local_space> array, ::sycl::nd_item<1> &item)
    {
        auto workItemIdx = item.get_local_id(0);
        auto sg = item.get_sub_group();
        auto workGroup = item.get_group();
        auto groupDim = item.get_local_range(0);

        // Comment out the first two lines of shift_left for Intel platform (min blockSize is 8 in that case)
        array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 16);
        array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 8);
        array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 4);
        array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 2);
        array[workItemIdx] += ::sycl::shift_group_left(sg, array[workItemIdx], 1);

        ::sycl::group_barrier(workGroup);

        if (workItemIdx == 0) {
            for (int i = 1; i < groupDim / 32; i++) {
                array[workItemIdx] += array[i * 32];
            }
        }
    }

    /// Get index of header vector of event data container for a given block ID.
    ///
    /// @param jag_vec the item jagged vector of edm
    /// @param header_idx the header idx
    /// @param ref_block_idx the reference block idx for a given header idx
    template <typename T>
    static void find_idx_on_jagged_vector(
        const vecmem::jagged_device_vector<T>& jag_vec,
        unsigned int& header_idx, unsigned int& item_idx, ::sycl::nd_item<1>& item) {

        // Equivalent to blockIdx.x in cuda
        auto groupIdx = item.get_group(0);
        // Equivalent to blockDim.x in cuda
        auto groupDim = item.get_local_range(0);
        // Equivalent to threadIdx.x in cuda
        auto workItemIdx = item.get_local_id(0);

        unsigned int ref_block_idx = 0;

        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_acc = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < jag_vec.size(); ++i) {
            nblocks_per_header = jag_vec[i].size() / groupDim + 1;
            nblocks_acc += nblocks_per_header;

            if (groupIdx < nblocks_acc) {
                header_idx = i;

                break;
            }
            ref_block_idx += nblocks_per_header;
        }
        item_idx = (groupIdx - ref_block_idx) * groupDim + workItemIdx;
    }
    /// Get index of header vector of event data container for a given block ID.
    ///
    /// @param container event data container where header element indicates the
    /// number of elements in item vector
    /// @param header_idx the header idx
    /// @param ref_block_idx the reference block idx for a given header idx
    template <typename header_t, typename item_t>
    static void find_idx_on_container(
        const device_container<header_t, item_t>& container,
        unsigned int& header_idx, unsigned int& item_idx, ::sycl::nd_item<1>& item) {

        // Equivalent to blockIdx.x in cuda
        auto groupIdx = item.get_group(0);
        // Equivalent to blockDim.x in cuda
        auto groupDim = item.get_local_range(0);
        // Equivalent to threadIdx.x in cuda
        auto workItemIdx = item.get_local_id(0);

        unsigned int ref_block_idx = 0;

        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < container.size(); ++i) {
            nblocks_per_header = container.get_headers()[i] / groupDim + 1;
            nblocks_accum += nblocks_per_header;

            if (groupIdx < nblocks_accum) {
                header_idx = i;

                break;
            }

            ref_block_idx += nblocks_per_header;
        }
        item_idx = (groupIdx - ref_block_idx) * groupDim + workItemIdx;
    }
};

}  // namespace sycl
}  // namespace traccc

