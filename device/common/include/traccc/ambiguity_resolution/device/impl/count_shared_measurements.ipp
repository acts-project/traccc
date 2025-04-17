/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE inline void count(
    const global_index_t globalIndex,
    const count_shared_measurements_payload& payload) {

    vecmem::device_vector<const unsigned int> accepted(payload.accepted_view);

    if (globalIndex >= accepted.size()) {
        return;
    }

    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::jagged_device_vector<const std::size_t> tracks_per_measurement(
        payload.tracks_per_measurement_view);
    vecmem::device_vector<unsigned int> shared(shared_view);

    const unsigned int id = accepted.at(globalIndex);
}

}  // namespace traccc::device