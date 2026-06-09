/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Compute the per-eta-bin minimum and maximum radius.
///
/// One thread per eta-bin scans the bin's node range and writes the (rmin,
/// rmax) pair into d_bin_rads_view; the host uses these to estimate the
/// maximum delta-R for each bin pair.
///
/// @param[in]  globalIndex             Current thread index
/// @param[in]  d_eta_bin_views_view    Per-eta (begin, end) node range
/// @param[in]  d_node_params_view      Sorted node geometry tuples (r is read)
/// @param[out] d_bin_rads_view         Per-eta (rmin, rmax) pair
///
TRACCC_HOST_DEVICE
inline void minmax_rad(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_eta_bin_views_view,
    const collection_types<float4>::const_view& d_node_params_view,
    const collection_types<float>::view& d_bin_rads_view);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/minmax_rad.ipp"
