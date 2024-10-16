/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/edm/track_candidate.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE inline void fill_sort_keys(
    std::size_t globalIndex, const fill_sort_keys_payload& payload) {

    bound_track_parameters_collection_types::const_device params(
        payload.params_view);

    // Keys
    vecmem::device_vector<device::sort_key> keys_device(payload.keys_view);

    // Param id
    vecmem::device_vector<unsigned int> ids_device(payload.ids_view);

    if (globalIndex >= keys_device.size()) {
        return;
    }

    keys_device.at(static_cast<unsigned int>(globalIndex)) =
        device::get_sort_key(params.at(static_cast<unsigned int>(globalIndex)));
    ids_device.at(static_cast<unsigned int>(globalIndex)) =
        static_cast<unsigned int>(globalIndex);
}

}  // namespace traccc::device
