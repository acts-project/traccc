/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/seed.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/seed_selecting_helper.hpp>

#include "utils/algorithm.hpp"

namespace traccc {

/// Seed filtering to filter out the bad triplets
struct seed_filtering
    : public algorithm<
          std::pair<host_triplet_collection&, host_seed_container&>(
              const host_spacepoint_container&, const sp_grid&)> {
    seed_filtering() {}

    output_type operator()(const host_spacepoint_container&,
                           const sp_grid&) const override {
        // not used
        __builtin_unreachable();
    }

    /// Callable operator for the seed filtering
    ///
    /// @param isp_container is internal spacepoint container
    /// @param triplets is the vector of triplets per middle spacepoint
    ///
    /// void interface
    ///
    /// @return seeds are the vector of seeds where the new compatible seeds are
    /// added
    void operator()(const host_spacepoint_container& sp_container,
                    const sp_grid& g2, output_type& o) const {
        auto& triplets = o.first;
        auto& seeds = o.second;

        host_seed_collection seeds_per_spM;

        for (auto& triplet : triplets) {
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

            if (!seed_selecting_helper::single_seed_cut(
                    m_filter_config, spM, spB, spT, triplet.weight)) {
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
                          auto& spB1 = sp_container.at(seed1.spB_link);
                          auto& spT1 = sp_container.at(seed1.spT_link);
                          auto& spB2 = sp_container.at(seed2.spB_link);
                          auto& spT2 = sp_container.at(seed2.spT_link);

                          seed1_sum += pow(spB1.y(), 2) + pow(spB1.z(), 2);
                          seed1_sum += pow(spT1.y(), 2) + pow(spT1.z(), 2);
                          seed2_sum += pow(spB2.y(), 2) + pow(spB2.z(), 2);
                          seed2_sum += pow(spT2.y(), 2) + pow(spT2.z(), 2);

                          return seed1_sum > seed2_sum;
                      }
                  });

        host_seed_collection new_seeds;
        if (seeds_per_spM.size() > 1) {
            new_seeds.push_back(seeds_per_spM[0]);

            size_t itLength = std::min(seeds_per_spM.size(),
                                       m_filter_config.max_triplets_per_spM);
            // don't cut first element
            for (size_t i = 1; i < itLength; i++) {
                if (seed_selecting_helper::cut_per_middle_sp(
                        m_filter_config, sp_container, seeds_per_spM[i],
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
            // seeds.push_back(*it);
            seeds.get_headers()[0]++;
            seeds.get_items()[0].push_back(*it);
        }
    }

    private:
    seedfilter_config m_filter_config;
};

}  // namespace traccc
