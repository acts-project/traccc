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
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void add_terminus_to_path_store(
    const global_index_t globalIndex,
    const collection_types<int2>::view d_path_store_view,
    const collection_types<short2>::const_view& d_outgoing_paths_view) {
    collection_types<int2>::device d_path_store(d_path_store_view);
    const collection_types<short2>::const_device d_outgoing_paths(
        d_outgoing_paths_view);

    const short2 out_paths = d_outgoing_paths[globalIndex];
    if (out_paths.y == -1) {
        return;
    }
    d_path_store[static_cast<unsigned int>(out_paths.y)] =
        make_int2(static_cast<int>(globalIndex), -1);
}

}  // namespace traccc::device
