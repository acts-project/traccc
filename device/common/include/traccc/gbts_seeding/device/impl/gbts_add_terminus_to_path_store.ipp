/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void gbts_add_terminus_to_path_store(
    const global_index_t globalIndex,
    const gbts_add_terminus_to_path_store_payload& payload) {

    vecmem::device_vector<int2> d_path_store(payload.path_store);
    const vecmem::device_vector<const short2> d_outgoing_paths(
        payload.outgoing_paths);

    const short2 out_paths = d_outgoing_paths[globalIndex];
    if (out_paths.y == -1) {
        return;
    }
    d_path_store[static_cast<unsigned int>(out_paths.y)] =
        int2{static_cast<int>(globalIndex), -1};
}

}  // namespace traccc::device
