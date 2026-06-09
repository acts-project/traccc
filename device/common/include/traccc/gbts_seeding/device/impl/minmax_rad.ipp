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
inline void minmax_rad(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_eta_bin_views_view,
    const collection_types<float4>::const_view& d_node_params_view,
    const collection_types<float>::view& d_bin_rads_view) {

    const collection_types<unsigned int>::const_device d_eta_bin_views(
        d_eta_bin_views_view);
    const collection_types<float4>::const_device d_node_params(
        d_node_params_view);
    collection_types<float>::device d_bin_rads(d_bin_rads_view);

    const unsigned int node_start = d_eta_bin_views[2u * globalIndex];
    const unsigned int node_end = d_eta_bin_views[2u * globalIndex + 1u];

    float min_r = 1e8f;
    float max_r = -1e8f;

    for (unsigned int node_idx = node_start; node_idx < node_end; node_idx++) {
        const float r = d_node_params[node_idx].z;
        max_r = fmaxf(r, max_r);
        min_r = fminf(r, min_r);
    }

    d_bin_rads[2u * globalIndex] = min_r;
    d_bin_rads[2u * globalIndex + 1u] = max_r;
}

}  // namespace traccc::device
