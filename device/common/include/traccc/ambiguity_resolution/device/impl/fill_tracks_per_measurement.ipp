/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE inline void fill_tracks_per_measurement(
    const global_index_t globalIndex,
    const fill_tracks_per_measurement_payload& payload) {

    vecmem::device_vector<const unsigned int> accepted(payload.accepted_view);

    if (globalIndex >= accepted.size()) {
        return;
    }

    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::jagged_device_vector<std::size_t> tracks_per_measurement(
        payload.tracks_per_measurement_view);

    const unsigned int id = accepted.at(globalIndex);

    const auto& candidates = track_candidates.at(id).items;
    for (const auto& cand : candidates) {
        const auto it =
            thrust::lower_bound(thrust::seq, unique_meas.begin(),
                                unique_meas.end(), cand.measurement_id);
        assert(it != unique_meas.end());
        const std::size_t unique_meas_idx =
            static_cast<std::size_t>(thrust::distance(unique_meas.begin(), it));
        tracks_per_measurement.at(unique_meas_idx).push_back(id);
    }
}

}  // namespace traccc::device
