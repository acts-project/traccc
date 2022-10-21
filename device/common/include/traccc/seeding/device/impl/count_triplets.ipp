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
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        doublet_ps_view,
    const doublet_container_types::const_view mid_bot_doublet_view,
    const doublet_container_types::const_view mid_top_doublet_view,
    triplet_counter_container_types::view triplet_view) {

    // Create device copy of input parameters
    vecmem::device_vector<const prefix_sum_element_t> doublet_prefix_sum(
        doublet_ps_view);

    // Check if anything needs to be done.
    if (globalIndex >= doublet_prefix_sum.size()) {
        return;
    }

    // Create device copy of input parameters
    const doublet_container_types::const_device mid_bot_doublet_device(
        mid_bot_doublet_view);

    // Get ids
    const prefix_sum_element_t ps_idx = doublet_prefix_sum[globalIndex];
    const unsigned int bin_idx = ps_idx.first;
    const unsigned int item_idx = ps_idx.second;

    // Get current mid bottom doublet
    const doublet mid_bot =
        mid_bot_doublet_device.get_items().at(bin_idx).at(item_idx);

    // Create device copy of input parameters
    const doublet_container_types::const_device mid_top_doublet_device(
        mid_top_doublet_view);

    // Create device copy of output parameter
    triplet_counter_container_types::device triplet_counter(triplet_view);

    // Get all spacepoints
    const const_sp_grid_device internal_sp_device(sp_view);

    // Header of doublet: number of mid_top doublets per bin
    // Item of doublet: doublet objects per bin
    const vecmem::device_vector<const doublet> mid_top_doublets_per_bin =
        mid_top_doublet_device.get_items().at(bin_idx);

    // middle spacepoint indexes
    const unsigned int spM_bin = mid_bot.sp1.bin_idx;
    const unsigned int spM_idx = mid_bot.sp1.sp_idx;
    // middle spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spM =
        internal_sp_device.bin(spM_bin)[spM_idx];
    // bottom spacepoint indexes
    const unsigned int spB_bin = mid_bot.sp2.bin_idx;
    const unsigned int spB_idx = mid_bot.sp2.sp_idx;
    // bottom spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spB =
        internal_sp_device.bin(spB_bin)[spB_idx];

    // Apply the conformal transformation to middle-bot doublet
    traccc::lin_circle lb =
        doublet_finding_helper::transform_coordinates(spM, spB, true);

    // Calculate some physical quantities required for triplet compatibility
    // check
    scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
    scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2;
    scatteringInRegion2 *= config.sigmaScattering * config.sigmaScattering;

    // These two quantities are used as output parameters in
    // triplet_finding_helper::isCompatible but their values are irrelevant
    scalar curvature, impact_parameter;

    // find the reference (start) index of the mid-top doublet container
    // item vector, where the doublets are recorded
    const unsigned int mt_start_idx = mid_bot.m_mt_start_idx;
    const unsigned int mt_end_idx = mid_bot.m_mt_end_idx;

    // number of triplets per middle-bot doublet
    unsigned int num_triplets_per_mb = 0;

    // iterate over mid-top doublets
    for (unsigned int i = mt_start_idx; i < mt_end_idx; ++i) {
        const traccc::doublet mid_top_doublet = mid_top_doublets_per_bin[i];

        const unsigned int spT_bin = mid_top_doublet.sp2.bin_idx;
        const unsigned int spT_idx = mid_top_doublet.sp2.sp_idx;
        const traccc::internal_spacepoint<traccc::spacepoint> spT =
            internal_sp_device.bin(spT_bin)[spT_idx];

        // Apply the conformal transformation to middle-top doublet
        traccc::lin_circle lt =
            doublet_finding_helper::transform_coordinates(spM, spT, false);

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
        triplet_counter_header& header =
            triplet_counter.get_headers().at(bin_idx);
        vecmem::device_atomic_ref<unsigned int> nMidBot(header.m_nMidBot);
        nMidBot.fetch_add(1);
        vecmem::device_atomic_ref<unsigned int> nTriplets(header.m_nTriplets);
        const unsigned int posTriplets =
            nTriplets.fetch_add(num_triplets_per_mb);

        triplet_counter.get_items().at(bin_idx).push_back(
            {mid_bot, num_triplets_per_mb, mt_start_idx, mt_end_idx,
             posTriplets});
    }
}

}  // namespace traccc::device
