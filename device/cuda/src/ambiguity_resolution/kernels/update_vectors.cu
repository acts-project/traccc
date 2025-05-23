/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "update_vectors.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::cuda::kernels {

__global__ void update_vectors(device::update_vectors_payload payload) {

    __shared__ unsigned int shared_tids[1024];
    __shared__ std::size_t shared_meas_ids[1024];

    if (*(payload.terminate) == 1) {
        return;
    }

    auto globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    auto threadIndex = threadIdx.x;

    shared_tids[globalIndex] = std::numeric_limits<unsigned int>::max();

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> n_meas(payload.n_meas_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::jagged_device_vector<const std::size_t> tracks_per_measurement(
        payload.tracks_per_measurement_view);
    vecmem::jagged_device_vector<int> track_status_per_measurement(
        payload.track_status_per_measurement_view);
    vecmem::device_vector<unsigned int> n_accepted_tracks_per_measurement(
        payload.n_accepted_tracks_per_measurement_view);
    vecmem::device_vector<unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<traccc::scalar> rel_shared(payload.rel_shared_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);
    vecmem::device_vector<int> is_updated(payload.is_updated_view);

    if (*payload.n_accepted == 0) {
        return;
    }

    const auto worst_track = sorted_ids[*payload.n_accepted - 1];
    const auto& worst_meas_list = meas_ids[worst_track];
    if (globalIndex < n_meas[worst_track]) {
        shared_meas_ids[globalIndex] = worst_meas_list[globalIndex];
    }

    if (globalIndex == 0) {
        (*payload.n_accepted)--;
    }

    __syncthreads();

    if (globalIndex >= n_meas[worst_track]) {
        return;
    }

    __syncthreads();

    const auto id = shared_meas_ids[globalIndex];

    bool is_duplicate = false;
    for (unsigned int i = 0; i < globalIndex; ++i) {
        if (shared_meas_ids[i] == id) {
            is_duplicate = true;
            break;
        }
    }
    if (is_duplicate) {
        return;
    }

    const std::size_t unique_meas_idx =
        thrust::lower_bound(thrust::seq, unique_meas.begin(), unique_meas.end(),
                            id) -
        unique_meas.begin();

    vecmem::device_atomic_ref<unsigned int> n_accepted_per_meas(
        n_accepted_tracks_per_measurement.at(
            static_cast<unsigned int>(unique_meas_idx)));
    const unsigned int N_A = n_accepted_per_meas.fetch_add(-1u);

    // If there is only one track associated with measurement, the
    // number of shared measurement can be reduced by one
    const auto& tracks = tracks_per_measurement[unique_meas_idx];
    auto track_status = track_status_per_measurement[unique_meas_idx];

    const unsigned int worst_idx =
        thrust::find(thrust::seq, tracks.begin(), tracks.end(), worst_track) -
        tracks.begin();

    track_status[worst_idx] = 0;

    if (N_A != 2) {
        return;
    }

    const unsigned int alive_idx =
        thrust::find(thrust::seq, track_status.begin(), track_status.end(), 1) -
        track_status.begin();

    shared_tids[threadIndex] = static_cast<unsigned int>(tracks[alive_idx]);
    auto tid = shared_tids[threadIndex];

    __syncthreads();

    const auto m_count = static_cast<unsigned int>(thrust::count(
        thrust::seq, meas_ids[tid].begin(), meas_ids[tid].end(), id));

    const unsigned int N_S =
        vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
            .fetch_sub(m_count);

    __syncthreads();

    bool already_pushed = false;
    for (unsigned int i = 0; i < threadIndex; ++i) {
        if (shared_tids[i] == tid) {
            already_pushed = true;
            break;
        }
    }
    if (!already_pushed) {

        // Write updated track IDs
        vecmem::device_atomic_ref<unsigned int> num_updated_tracks(
            *(payload.n_updated_tracks));

        const unsigned int pos = num_updated_tracks.fetch_add(1);

        updated_tracks[pos] = tid;
        is_updated[tid] = 1;

        rel_shared.at(tid) = static_cast<traccc::scalar>(n_shared.at(tid)) /
                             static_cast<traccc::scalar>(n_meas.at(tid));
    }
}

}  // namespace traccc::cuda::kernels
