/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/seed_filtering.hpp"

#include "traccc/seeding/seed_selecting_helper.hpp"

namespace traccc {

seed_filtering::seed_filtering(const seedfilter_config& config)
    : m_filter_config(config) {}

void seed_filtering::operator()(
    const spacepoint_collection_types::host& sp_collection, const sp_grid& g2,
    triplet_collection_types::host& triplets,
    seed_collection_types::host& seeds) const {

    seed_collection_types::host seeds_per_spM;

    for (triplet& triplet : triplets) {
        // bottom
        const auto& spB_idx = triplet.sp1;
        const auto& spB = g2.bin(spB_idx.bin_idx)[spB_idx.sp_idx];

        // middle
        const auto& spM_idx = triplet.sp2;
        const auto& spM = g2.bin(spM_idx.bin_idx)[spM_idx.sp_idx];

        // top
        const auto& spT_idx = triplet.sp3;
        const auto& spT = g2.bin(spT_idx.bin_idx)[spT_idx.sp_idx];

        seed_selecting_helper::seed_weight(m_filter_config, spM, spB, spT,
                                           triplet.weight);

        if (!seed_selecting_helper::single_seed_cut(m_filter_config, spM, spB,
                                                    spT, triplet.weight)) {
            continue;
        }

        seeds_per_spM.push_back({spB.m_link, spM.m_link, spT.m_link,
                                 triplet.weight, triplet.z_vertex});
    }

    // sort seeds based on their weights
    std::sort(seeds_per_spM.begin(), seeds_per_spM.end(),
              [&](seed& seed1, seed& seed2) {
                  if (seed1.weight != seed2.weight) {
                      return seed1.weight > seed2.weight;
                  } else {
                      scalar seed1_sum = 0;
                      scalar seed2_sum = 0;
                      auto& spB1 = sp_collection.at(seed1.spB_link);
                      auto& spT1 = sp_collection.at(seed1.spT_link);
                      auto& spB2 = sp_collection.at(seed2.spB_link);
                      auto& spT2 = sp_collection.at(seed2.spT_link);

                      seed1_sum += pow(spB1.y(), 2) + pow(spB1.z(), 2);
                      seed1_sum += pow(spT1.y(), 2) + pow(spT1.z(), 2);
                      seed2_sum += pow(spB2.y(), 2) + pow(spB2.z(), 2);
                      seed2_sum += pow(spT2.y(), 2) + pow(spT2.z(), 2);

                      return seed1_sum > seed2_sum;
                  }
              });

    seed_collection_types::host new_seeds;
    if (seeds_per_spM.size() > 1) {
        new_seeds.push_back(seeds_per_spM[0]);

        size_t itLength = std::min(seeds_per_spM.size(),
                                   m_filter_config.max_triplets_per_spM);
        // don't cut first element
        for (size_t i = 1; i < itLength; i++) {
            if (seed_selecting_helper::cut_per_middle_sp(
                    m_filter_config, sp_collection, seeds_per_spM[i],
                    seeds_per_spM[i].weight)) {
                new_seeds.push_back(std::move(seeds_per_spM[i]));
            }
        }
        seeds_per_spM = std::move(new_seeds);
    }

    unsigned int maxSeeds = seeds_per_spM.size();

    if (maxSeeds > m_filter_config.maxSeedsPerSpM) {
        maxSeeds = m_filter_config.maxSeedsPerSpM + 1;
    }

    auto itBegin = seeds_per_spM.begin();
    auto it = seeds_per_spM.begin();
    // default filter removes the last seeds if maximum amount exceeded
    // ordering by weight by filterSeeds_2SpFixed means these are the lowest
    // weight seeds

    for (; it < itBegin + maxSeeds; ++it) {
        seeds.push_back(*it);
    }
}

}  // namespace traccc
