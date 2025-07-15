/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "find_max_shared.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void find_max_shared(device::find_max_shared_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<int> is_updated(payload.is_updated_view);

    auto globalIndex = details::global_index1();
    if (globalIndex < is_updated.size()) {
        is_updated[globalIndex] = 0;
    }

    if (globalIndex >= *payload.n_accepted) {
        return;
    }

    auto tid = sorted_ids[globalIndex];
    auto shared = n_shared[tid];

    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned int other_shared =
            __shfl_down_sync(0xffffffff, shared, offset);

        if (other_shared > shared) {
            shared = other_shared;
        }
    }

    if (threadIdx.x == 0) {
        atomicMax(payload.max_shared, shared);
    }
}

}  // namespace traccc::cuda::kernels
