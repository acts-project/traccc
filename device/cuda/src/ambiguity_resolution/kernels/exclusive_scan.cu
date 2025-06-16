/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "exclusive_scan.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void exclusive_scan(device::exclusive_scan_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    __shared__ int prefix[1024];
    __shared__ measurement_id_type sh_meas_ids[1024];
    __shared__ unsigned int sh_threads[1024];

    auto threadIndex = threadIdx.x;

    if (threadIndex >= *(payload.n_meas_to_remove)) {
        return;
    }

    vecmem::device_vector<measurement_id_type> meas_to_remove(
        payload.meas_to_remove_view);
    vecmem::device_vector<unsigned int> threads(payload.threads_view);

    auto n_meas_to_remove_temp = *(payload.n_meas_to_remove);

    if (threadIndex == 0) {
        *(payload.n_meas_to_remove) = 0;
    }

    __syncthreads();

    int is_valid =
        (threads[threadIndex] < *(payload.n_removable_tracks)) ? 1 : 0;

    // TODO: Use better reduction algorithm
    if (is_valid) {
        atomicAdd(payload.n_meas_to_remove, 1);
    }

    __syncthreads();

    // Exclusive scan (Hillis-Steele)
    prefix[threadIndex] = is_valid;  // copy input
    __syncthreads();

    for (int offset = 1; offset < n_meas_to_remove_temp; offset <<= 1) {
        int val = 0;
        if (threadIndex >= offset) {
            val = prefix[threadIndex - offset];
        }
        __syncthreads();
        prefix[threadIndex] += val;
        __syncthreads();
    }

    if (is_valid) {
        prefix[threadIndex] -= 1;
        sh_meas_ids[prefix[threadIndex]] = meas_to_remove[threadIndex];
        sh_threads[prefix[threadIndex]] = threads[threadIndex];
    }

    __syncthreads();

    meas_to_remove[threadIndex] = sh_meas_ids[threadIndex];
    threads[threadIndex] = sh_threads[threadIndex];
}

}  // namespace traccc::cuda::kernels
