/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <edm/seed.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/seed_selecting_helper.hpp>

#pragma once

namespace traccc {

/// Seed filtering to filter out the bad triplets
struct seed_filtering {
    seed_filtering() {}

    /// Callable operator for the seed filtering
    ///
    /// @param isp_container is internal spacepoint container
    /// @param triplets is the vector of triplets per middle spacepoint
    ///
    /// void interface
    ///
    /// @return seeds are the vector of seeds where the new compatible seeds are
    /// added
    void operator()(const host_internal_spacepoint_container& isp_container,
                    host_triplet_collection& triplets,
                    host_seed_container& seeds) {
        host_seed_collection seeds_per_spM;

        for (auto& triplet : triplets) {
            // bottom
            auto spB_idx = triplet.sp1;
            auto spB = isp_container.items[spB_idx.bin_idx][spB_idx.sp_idx];

            // middle
            auto spM_idx = triplet.sp2;
            auto spM = isp_container.items[spM_idx.bin_idx][spM_idx.sp_idx];

            // top
            auto spT_idx = triplet.sp3;
            auto spT = isp_container.items[spT_idx.bin_idx][spT_idx.sp_idx];

            seed_selecting_helper::seed_weight(m_filter_config, spM, spB, spT,
                                               triplet.weight);

            if (!seed_selecting_helper::single_seed_cut(
                    m_filter_config, spM, spB, spT, triplet.weight)) {
                continue;
            }

            seeds_per_spM.push_back({spB.m_sp, spM.m_sp, spT.m_sp,
                                     triplet.weight, triplet.z_vertex});
        }

        // sort seeds based on their weights
        std::sort(seeds_per_spM.begin(), seeds_per_spM.end(),
                  [](seed& seed1, seed& seed2) {
                      if (seed1.weight != seed2.weight) {
                          return seed1.weight > seed2.weight;
                      } else {
                          return std::abs(seed1.z_vertex) <
                                 std::abs(seed2.z_vertex);
                          /*
                          float seed1_sum = 0;
                          float seed2_sum = 0;
                          seed1_sum += pow(seed1.spB.y(),2) +
                          pow(seed1.spB.z(),2); seed1_sum +=
                          pow(seed1.spM.y(),2) + pow(seed1.spM.z(),2); seed1_sum
                          += pow(seed1.spT.y(),2) + pow(seed1.spT.z(),2);

                          seed2_sum += pow(seed2.spB.y(),2) +
                          pow(seed2.spB.z(),2); seed2_sum +=
                          pow(seed2.spM.y(),2) + pow(seed2.spM.z(),2); seed2_sum
                          += pow(seed2.spT.y(),2) + pow(seed2.spT.z(),2);

                          return seed1_sum > seed2_sum;
                          */
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
                        m_filter_config, seeds_per_spM[i].spM,
                        seeds_per_spM[i].spB, seeds_per_spM[i].spT,
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
            seeds.headers[0]++;
            seeds.items[0].push_back(*it);
        }
    }

    private:
    seedfilter_config m_filter_config;
};

}  // namespace traccc
