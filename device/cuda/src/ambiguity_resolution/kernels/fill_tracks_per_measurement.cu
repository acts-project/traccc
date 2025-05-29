/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "fill_tracks_per_measurement.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::cuda::kernels {

__global__ void fill_tracks_per_measurement(
    device::fill_tracks_per_measurement_payload payload) {

    vecmem::device_vector<const unsigned int> accepted_ids(
        payload.accepted_ids_view);

    const auto globalIndex = details::global_index1();
    if (globalIndex >= accepted_ids.size()) {
        return;
    }

    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::jagged_device_vector<std::size_t> tracks_per_measurement(
        payload.tracks_per_measurement_view);
    vecmem::jagged_device_vector<int> track_status_per_measurement(
        payload.track_status_per_measurement_view);
    vecmem::device_vector<unsigned int> n_accepted_tracks_per_measurement(
        payload.n_accepted_tracks_per_measurement_view);

    const unsigned int id = accepted_ids.at(globalIndex);

    for (unsigned int i = 0; i < meas_ids[id].size(); i++) {
        auto meas_id = meas_ids[id][i];

        if (thrust::find(thrust::seq, meas_ids[id].begin(),
                         meas_ids[id].begin() + i,
                         meas_id) != (meas_ids[id].begin() + i)) {
            continue;
        }

        const auto it = thrust::lower_bound(thrust::seq, unique_meas.begin(),
                                            unique_meas.end(), meas_id);
        const std::size_t unique_meas_idx =
            static_cast<std::size_t>(thrust::distance(unique_meas.begin(), it));

        auto tracks = tracks_per_measurement.at(unique_meas_idx);

        tracks_per_measurement.at(unique_meas_idx).push_back(id);
        track_status_per_measurement.at(unique_meas_idx).push_back(1);

        vecmem::device_atomic_ref<unsigned int> n_accepted(
            n_accepted_tracks_per_measurement.at(
                static_cast<unsigned int>(unique_meas_idx)));
        n_accepted.fetch_add(1u);
    }
}
}  // namespace traccc::cuda::kernels
