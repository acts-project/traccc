/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "traccc/seeding/doublet_finding_helper.hpp"
#include "traccc/seeding/triplet_finding_helper.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_triplets(
    const std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid_const_view& sp_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& mb_ps_view,
    const doublet_spM_container_types::const_view mid_bot_doublet_view,
    const doublet_spM_container_types::const_view mid_top_doublet_view,
    triplet_counter_spM_container_types::view tc_view) {

    // Create device copy of input parameters
    vecmem::device_vector<const prefix_sum_element_t> mb_prefix_sum(mb_ps_view);

    // Check if anything needs to be done.
    if (globalIndex >= mb_prefix_sum.size()) {
        return;
    }

    // Create device copy of input parameters
    const doublet_spM_container_types::const_device mid_bot_doublet_device(
        mid_bot_doublet_view);

    // Get ids
    const prefix_sum_element_t ps_idx = mb_prefix_sum[globalIndex];
    const unsigned int bin_idx = ps_idx.first;
    const unsigned int item_idx = ps_idx.second;

    const traccc::sp_location spM_loc =
        mid_bot_doublet_device.get_headers().at(bin_idx).spM;
    const traccc::sp_location spB_loc =
        mid_bot_doublet_device.get_items().at(bin_idx).at(item_idx).sp2;

    // Create device copy of input parameters
    const doublet_spM_container_types::const_device mid_top_doublet_device(
        mid_top_doublet_view);

    // Get internal spacepoints for current bin
    const unsigned int num_top =
        mid_top_doublet_device.get_headers().at(bin_idx).n_doublets;

    // Create device copy of output parameter
    triplet_counter_spM_container_types::device triplet_counter(tc_view);

    // Get all spacepoints
    const const_sp_grid_device internal_sp_device(sp_view);

    // Header of doublet: number of mid_top doublets per bin
    // Item of doublet: doublet objects per bin
    const doublet_spM_collection_types::const_device mid_top_doublets_per_bin =
        mid_top_doublet_device.get_items().at(bin_idx);

    // middle spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spM =
        internal_sp_device.bin(spM_loc.bin_idx)[spM_loc.sp_idx];
    // bottom spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spB =
        internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];

    // Apply the conformal transformation to middle-bot doublet
    traccc::lin_circle lb = doublet_finding_helper::transform_coordinates<
        details::spacepoint_type::bottom>(spM, spB);

    // Calculate some physical quantities required for triplet compatibility
    // check
    scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
    scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2;
    scatteringInRegion2 *= config.sigmaScattering * config.sigmaScattering;

    // These two quantities are used as output parameters in
    // triplet_finding_helper::isCompatible but their values are irrelevant
    scalar curvature, impact_parameter;

    // number of triplets per middle-bot doublet
    unsigned int num_triplets_per_mb = 0;

    triplet_counter_spM_header& header =
        triplet_counter.get_headers().at(bin_idx);
    if (item_idx == 0) {
        header.m_spM = spM_loc;
    }

    // iterate over mid-top doublets
    for (unsigned int i = 0; i < num_top; ++i) {
        const traccc::sp_location spT_loc = mid_top_doublets_per_bin[i].sp2;

        const traccc::internal_spacepoint<traccc::spacepoint> spT =
            internal_sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

        // Apply the conformal transformation to middle-top doublet
        traccc::lin_circle lt = doublet_finding_helper::transform_coordinates<
            details::spacepoint_type::top>(spM, spT);

        // Check if mid-bot and mid-top doublets can form a triplet
        if (triplet_finding_helper::isCompatible(
                spM, lb, lt, config, iSinTheta2, scatteringInRegion2, curvature,
                impact_parameter)) {
            num_triplets_per_mb++;
        }
    }

    // if the number of triplets per mb is larger than 0, write the triplet
    // counter into the container
    if (num_triplets_per_mb > 0) {
        vecmem::device_atomic_ref<unsigned int> nTriplets(header.m_nTriplets);
        const unsigned int posTriplets =
            nTriplets.fetch_add(num_triplets_per_mb);

        triplet_counter.get_items().at(bin_idx).push_back(
            {spB_loc, num_triplets_per_mb, posTriplets});
    }
}

}  // namespace traccc::device
