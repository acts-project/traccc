/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "block_inclusive_scan.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void block_inclusive_scan(
    device::block_inclusive_scan_payload payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    extern __shared__ int shared_temp[];

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const int> is_updated(payload.is_updated_view);
    vecmem::device_vector<int> block_offsets(payload.block_offsets_view);
    vecmem::device_vector<int> prefix_sums(payload.prefix_sums_view);

    auto globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    auto threadIndex = threadIdx.x;

    const unsigned int n_accepted = *(payload.n_accepted);

    if (globalIndex >= n_accepted) {
        shared_temp[threadIndex] = 0;
    } else {
        shared_temp[threadIndex] = is_updated[sorted_ids[globalIndex]];
    }

    __syncthreads();

    // inclusive scan in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = 0;
        if (threadIndex >= stride) {
            val = shared_temp[threadIndex - stride];
        }
        __syncthreads();
        if (threadIndex >= stride) {
            shared_temp[threadIndex] += val;
        }
        __syncthreads();
    }

    if (globalIndex < n_accepted) {
        prefix_sums[globalIndex] = shared_temp[threadIndex];
    }

    __syncthreads();

    if (threadIndex == blockDim.x - 1) {
        block_offsets[blockIdx.x] = shared_temp[threadIndex];
    }
}

}  // namespace traccc::cuda::kernels
