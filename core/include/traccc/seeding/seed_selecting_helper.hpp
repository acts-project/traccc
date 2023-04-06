/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"

namespace traccc {

// helper function used for both cpu and gpu
struct seed_selecting_helper {
    /// Update the weights of triplets
    ///
    /// @param filter_config is seed filtering configuration parameters
    /// @param spM is middle (internal) spacepoint
    /// @param spB is bottom (internal) spacepoint
    /// @param spT is top (internal) spacepoint
    /// @param triplet_weight is the weight of triplet to be updated
    static TRACCC_HOST_DEVICE void seed_weight(
        const seedfilter_config& filter_config,
        const internal_spacepoint<spacepoint>&,
        const internal_spacepoint<spacepoint>& spB,
        const internal_spacepoint<spacepoint>& spT, scalar& triplet_weight) {
        scalar weight = 0;

        if (spB.radius() > filter_config.good_spB_min_radius) {
            weight = filter_config.good_spB_weight_increase;
        }
        if (spT.radius() < filter_config.good_spT_max_radius) {
            weight = filter_config.good_spT_weight_increase;
        }

        triplet_weight += weight;
        return;
    }

    /// Cut triplets with criteria
    ///
    /// @param filter_config is seed filtering configuration parameters
    /// @param spM is middle (internal) spacepoint
    /// @param spB is bottom (internal) spacepoint
    /// @param spT is top (internal) spacepoint
    /// @param triplet_weight is the weight of triplet
    ///
    /// @return boolean value
    static TRACCC_HOST_DEVICE bool single_seed_cut(
        const seedfilter_config& filter_config,
        const internal_spacepoint<spacepoint>&,
        const internal_spacepoint<spacepoint>& spB,
        const internal_spacepoint<spacepoint>&, const scalar& triplet_weight) {
        return !(spB.radius() > filter_config.good_spB_min_radius &&
                 triplet_weight < filter_config.good_spB_min_weight);
    }

    /// Cut triplets with criteria
    ///
    /// @param filter_config    seed filtering configuration parameters
    /// @param sp_collection    spacepoint collection
    /// @param seed             current seed to possibly cut
    /// @param triplet_weight   triplets' weight
    ///
    /// @return boolean value
    template <typename spacepoint_collection_t>
    static TRACCC_HOST_DEVICE bool cut_per_middle_sp(
        const seedfilter_config& filter_config,
        const spacepoint_collection_t& sp_collection, const seed& seed,
        const scalar& triplet_weight) {

        const auto& spB = sp_collection.at(seed.spB_link);

        return (triplet_weight > filter_config.seed_min_weight ||
                spB.radius() > filter_config.spB_min_radius);
    }
};

}  // namespace traccc
