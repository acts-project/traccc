/** traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "gather_measurement_votes.cuh"

namespace traccc::cuda::kernels {

__global__ void gather_measurement_votes(
    const vecmem::data::vector_view<const unsigned long long int>
        insertion_mutex_view,
    const vecmem::data::vector_view<const unsigned int> tip_index_view,
    vecmem::data::vector_view<unsigned int> votes_per_tip_view,
    const unsigned int max_num_tracks_per_measurement) {
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int measurement_idx = thread_idx / max_num_tracks_per_measurement;
    unsigned int tip_idx = thread_idx % max_num_tracks_per_measurement;

    const vecmem::device_vector<const unsigned long long int> insertion_mutex(
        insertion_mutex_view);
    const vecmem::device_vector<const unsigned int> tip_index(tip_index_view);
    vecmem::device_vector<unsigned int> votes_per_tip(votes_per_tip_view);

    if (measurement_idx >= insertion_mutex.size()) {
        return;
    }

    auto [locked, size, worst] =
        device::decode_insertion_mutex(insertion_mutex.at(measurement_idx));

    if (tip_idx < size) {
        vecmem::device_atomic_ref<unsigned int>(
            votes_per_tip.at(tip_index.at(thread_idx)))
            .fetch_add(1u);
    }
}

}  // namespace traccc::cuda::kernels
