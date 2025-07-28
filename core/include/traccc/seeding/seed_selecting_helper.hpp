/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"

namespace traccc {

// helper function used for both cpu and gpu
struct seed_selecting_helper {
    /// Update the weights of triplets
    ///
    /// @param[in] filter_config is seed filtering configuration parameters
    /// @param[in] spM is middle (internal) spacepoint
    /// @param[in] spB is bottom (internal) spacepoint
    /// @param[in] spT is top (internal) spacepoint
    /// @param[out] triplet_weight is the weight of triplet to be updated
    ///
    template <typename T1, typename T2, typename T3>
    static TRACCC_HOST_DEVICE void seed_weight(
        const seedfilter_config& filter_config, const edm::spacepoint<T1>&,
        const edm::spacepoint<T2>& spB, const edm::spacepoint<T3>& spT,
        scalar& triplet_weight) {

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
    template <typename T1, typename T2, typename T3>
    static TRACCC_HOST_DEVICE bool single_seed_cut(
        const seedfilter_config& filter_config, const edm::spacepoint<T1>&,
        const edm::spacepoint<T2>& spB, const edm::spacepoint<T3>&,
        scalar triplet_weight) {

        return !(spB.radius() > filter_config.good_spB_min_radius &&
                 triplet_weight < filter_config.good_spB_min_weight);
    }
};

}  // namespace traccc
