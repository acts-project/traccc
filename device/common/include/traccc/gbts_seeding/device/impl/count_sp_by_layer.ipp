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
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// Detray include(s).
#include <detray/geometry/identifier.hpp>

// System include(s).
#include <array>
#include <climits>
#include <cstdint>
#include <utility>

namespace traccc::device {

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
    const unsigned long int volumeMapSize,
    const unsigned long int surfaceMapSize,
    const gbts_sp_counting_params sp_counting_params) {

    const traccc::edm::spacepoint_collection::const_device spacepoints(
        spacepoints_view);
    const edm::measurement_collection::const_device measurements(
        measurements_view);
    const collection_types<short>::const_device volumeToLayerMap(
        volumeToLayerMap_view);
    const collection_types<std::pair<unsigned int, unsigned int>>::const_device
        surfaceToLayerMap(surfaceToLayerMap_view);
    const collection_types<char>::const_device layerType(layerType_view);

    collection_types<unsigned int>::device layerCounts(layerCounts_view);
    collection_types<unsigned short>::device spacepointsLayer(
        spacepointsLayer_view);
    collection_types<float4>::device reducedSP(reducedSP_view);

    const auto spacepoint = spacepoints.at(globalIndex);
    const auto measurement = measurements.at(spacepoint.measurement_index_1());

    const detray::geometry::identifier geo_id = measurement.surface_link();
    const unsigned int volume = geo_id.volume();
    const short begin_or_bin =
        (volume < volumeMapSize) ? volumeToLayerMap[volume] : SHRT_MAX;

    if (begin_or_bin == SHRT_MAX) {
        reducedSP[globalIndex].w = -CHAR_MAX - 1;
        return;
    }
    unsigned int layerIdx = 0u;
    if (begin_or_bin < 0) {
        const unsigned int surface_index =
            static_cast<unsigned int>(geo_id.index());

        for (unsigned int surface = -1 * (begin_or_bin + 1);
             surface < surfaceMapSize; surface++) {

            const std::pair<unsigned int, unsigned int> surfaceBinPair =
                surfaceToLayerMap[surface];
            if (surfaceBinPair.first == surface_index) {
                layerIdx = surfaceBinPair.second;
                break;
            }
        }
    } else {
        layerIdx = static_cast<unsigned int>(begin_or_bin);
    }
    float cluster_diameter = measurement.diameter();
    const int type = static_cast<int>(layerType[layerIdx]);
    if (type == 1 && cluster_diameter > sp_counting_params.type1_max_width) {
        reducedSP[globalIndex].w = -CHAR_MAX - 1;
        return;
    }
    cluster_diameter = (sp_counting_params.doTauCut && type != 0)
                           ? static_cast<float>(-1 * type)
                           : cluster_diameter;

    vecmem::device_atomic_ref<unsigned int>(layerCounts[layerIdx])
        .fetch_add(1u);
    spacepointsLayer[globalIndex] = static_cast<unsigned short>(layerIdx);
    const std::array<float, 3u> pos = spacepoint.global();
    reducedSP[globalIndex] =
        make_float4(pos[0], pos[1], pos[2], cluster_diameter);
    // global x, y, z, and diameter
}

}  // namespace traccc::device
