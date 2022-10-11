/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/seed_finding.hpp"

namespace traccc {

seed_finding::seed_finding(const seedfinder_config& finder_config,
                           const seedfilter_config& filter_config)
    : m_doublet_finding(finder_config.toInternalUnits()),
      m_triplet_finding(finder_config.toInternalUnits()),
      m_seed_filtering(filter_config.toInternalUnits()) {}

seed_finding::output_type seed_finding::operator()(
    const spacepoint_container_types::host& sp_container,
    const sp_grid& g2) const {

    // Run the algorithm
    output_type seeds;

    const bool bottom = true;
    const bool top = false;

    for (unsigned int i = 0; i < g2.nbins(); i++) {
        auto& spM_collection = g2.bin(i);

        for (unsigned int j = 0; j < spM_collection.size(); ++j) {

            sp_location spM_location({i, j});

            // middule-bottom doublet search
            auto mid_bot = m_doublet_finding(g2, spM_location, bottom);

            if (mid_bot.first.empty())
                continue;

            // middule-top doublet search
            auto mid_top = m_doublet_finding(g2, spM_location, top);

            if (mid_top.first.empty())
                continue;

            triplet_collection_types::host triplets_per_spM;

            // triplet search from the combinations of two doublets which
            // share middle spacepoint
            for (unsigned int k = 0; k < mid_bot.first.size(); ++k) {
                auto& doublet_mb = mid_bot.first[k];
                auto& lb = mid_bot.second[k];

                triplet_collection_types::host triplets = m_triplet_finding(
                    g2, doublet_mb, lb, mid_top.first, mid_top.second);

                triplets_per_spM.insert(std::end(triplets_per_spM),
                                        triplets.begin(), triplets.end());
            }

            // seed filtering
            m_seed_filtering(sp_container, g2, triplets_per_spM, seeds);
        }
    }

    return seeds;
}

}  // namespace traccc
