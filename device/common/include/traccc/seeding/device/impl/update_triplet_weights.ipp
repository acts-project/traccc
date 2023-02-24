/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// System include(s).
#include <cassert>

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void update_triplet_weights(
    const std::size_t globalIndex, const seedfilter_config& filter_config,
    const sp_grid_const_view& sp_view,
    const triplet_counter_spM_collection_types::const_view& spM_tc_view,
    const triplet_counter_collection_types::const_view& tc_view, scalar* data,
    device_triplet_collection_types::view triplet_view) {

    // Check if anything needs to be done.
    device_triplet_collection_types::device triplets(triplet_view);
    if (globalIndex >= triplets.size()) {
        return;
    }

    // Set up the device containers
    const const_sp_grid_device sp_grid(sp_view);
    const triplet_counter_spM_collection_types::const_device triplet_counts_spM(
        spM_tc_view);
    const triplet_counter_collection_types::const_device triplet_counts(
        tc_view);

    // Current work item
    device_triplet this_triplet = triplets[globalIndex];

    const sp_location& spT_idx = this_triplet.spT;

    const traccc::internal_spacepoint<traccc::spacepoint> current_spT =
        sp_grid.bin(spT_idx.bin_idx)[spT_idx.sp_idx];

    const scalar currentTop_r = current_spT.radius();

    // if two compatible seeds with high distance in r are found, compatible
    // seeds span 5 layers
    // -> very good seed
    const scalar lowerLimitCurv =
        this_triplet.curvature - filter_config.deltaInvHelixDiameter;
    const scalar upperLimitCurv =
        this_triplet.curvature + filter_config.deltaInvHelixDiameter;
    std::size_t num_compat_seedR = 0;

    const triplet_counter mb_count =
        triplet_counts.at(this_triplet.counter_link);

    // Check if anything needs to be done.
    if (mb_count.m_nTriplets <= 1) {
        return;
    }

    // Get the position of the triplets which share this sabe midBottom doublet
    const unsigned int triplets_mb_begin =
        mb_count.posTriplets +
        triplet_counts_spM.at(mb_count.spM_counter_link).posTriplets;
    const unsigned int triplets_mb_end =
        triplets_mb_begin + mb_count.m_nTriplets;

    // iterate over triplets
    for (unsigned int i = triplets_mb_begin; i < triplets_mb_end; ++i) {
        // skip same triplet
        if (i == globalIndex) {
            continue;
        }

        const device_triplet other_triplet = triplets[i];
        const sp_location other_spT_idx = other_triplet.spT;
        const traccc::internal_spacepoint<traccc::spacepoint> other_spT =
            sp_grid.bin(other_spT_idx.bin_idx)[other_spT_idx.sp_idx];

        // compared top SP should have at least deltaRMin distance
        const scalar otherTop_r = other_spT.radius();
        const scalar deltaR = currentTop_r - otherTop_r;
        if (std::abs(deltaR) < filter_config.deltaRMin) {
            continue;
        }

        // curvature difference within limits?
        // TODO: how much slower than sorting all vectors by curvature
        // and breaking out of loop? i.e. is vector size large (e.g. in
        // jets?)
        if (other_triplet.curvature < lowerLimitCurv) {
            continue;
        }
        if (other_triplet.curvature > upperLimitCurv) {
            continue;
        }

        bool newCompSeed = true;

        for (std::size_t i_s = 0; i_s < num_compat_seedR; ++i_s) {
            const scalar previousDiameter = data[i_s];

            // original ATLAS code uses higher min distance for 2nd found
            // compatible seed (20mm instead of 5mm) add new compatible seed
            // only if distance larger than rmin to all other compatible
            // seeds
            if (std::abs(previousDiameter - otherTop_r) <
                filter_config.deltaRMin) {
                newCompSeed = false;
                break;
            }
        }

        if (newCompSeed) {
            data[num_compat_seedR] = otherTop_r;
            this_triplet.weight += filter_config.compatSeedWeight;
            num_compat_seedR++;
        }

        if (num_compat_seedR >= filter_config.compatSeedLimit) {
            break;
        }
    }

    triplets[globalIndex].weight = this_triplet.weight;
}

}  // namespace traccc::device
