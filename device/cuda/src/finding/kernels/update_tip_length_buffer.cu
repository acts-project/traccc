/** traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "update_tip_length_buffer.cuh"

namespace traccc::cuda::kernels {

__global__ void update_tip_length_buffer(
    const vecmem::data::vector_view<const unsigned int> old_tip_length_view,
    vecmem::data::vector_view<unsigned int> new_tip_length_view,
    const vecmem::data::vector_view<const unsigned int> measurement_votes_view,
    unsigned int* tip_to_output_map, unsigned int* tip_to_output_map_idx,
    float min_measurement_voting_fraction) {
    const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(tip_to_output_map != nullptr);
    assert(tip_to_output_map_idx != nullptr);

    const vecmem::device_vector<const unsigned int> old_tip_length(
        old_tip_length_view);
    vecmem::device_vector<unsigned int> new_tip_length(new_tip_length_view);
    const vecmem::device_vector<const unsigned int> measurement_votes(
        measurement_votes_view);

    if (thread_idx >= measurement_votes_view.size()) {
        return;
    }

    const unsigned int total_measurements = old_tip_length.at(thread_idx);
    const unsigned int total_votes = measurement_votes.at(thread_idx);

    assert(total_votes <= total_measurements);

    const scalar vote_fraction = static_cast<scalar>(total_votes) /
                                 static_cast<scalar>(total_measurements);

    if (vote_fraction < min_measurement_voting_fraction) {
        tip_to_output_map[thread_idx] =
            std::numeric_limits<unsigned int>::max();
    } else {
        const auto new_idx =
            vecmem::device_atomic_ref(*tip_to_output_map_idx).fetch_add(1u);
        new_tip_length.at(new_idx) = total_measurements;
        tip_to_output_map[thread_idx] = new_idx;
    }
}

}  // namespace traccc::cuda::kernels
