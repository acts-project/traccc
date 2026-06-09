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
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Count and tag spacepoints by GBTS layer, producing the reduced SP
/// view.
///
/// Each thread inspects one spacepoint, looks up its GBTS layer via the
/// volume / surface map, optionally applies a cluster-width cut, and on
/// acceptance atomically increments the per-layer count and writes the
/// reduced (x, y, z, r) tuple plus the assigned layer index.
///
/// @param[in]  globalIndex          Current thread index
/// @param[in]  spacepoints_view     All spacepoints in the event
/// @param[in]  measurements_view    All measurements (for surface lookup and
///                                  cluster width)
/// @param[in]  volumeToLayerMap_view Map from volume index to GBTS layer
/// @param[in]  surfaceToLayerMap_view Map from (volume, surface) to layer
///                                  (used when size > 0)
/// @param[in]  layerType_view       Per-layer barrel/endcap type code
/// @param[out] reducedSP_view       Reduced (x, y, z, r) per spacepoint
/// @param[out] layerCounts_view     Per-layer atomic spacepoint counter
/// @param[out] spacepointsLayer_view GBTS layer assigned to each kept SP
/// @param[in]  sp_counting_params   Parameters for spacepoint counting
///
TRACCC_HOST_DEVICE
inline void count_sp_by_layer(
    const global_index_t globalIndex,
    const traccc::edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::measurement_collection::const_view& measurements_view,
    const collection_types<short>::const_view& volumeToLayerMap_view,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view&
        surfaceToLayerMap_view,
    const collection_types<char>::const_view& layerType_view,
    const collection_types<float4>::view reducedSP_view,
    const collection_types<unsigned int>::view layerCounts_view,
    const collection_types<unsigned short>::view spacepointsLayer_view,
    const gbts_sp_counting_params sp_counting_params);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/count_sp_by_layer.ipp"
