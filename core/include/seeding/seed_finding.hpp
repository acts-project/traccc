/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <iostream>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/statistics.hpp>
#include <seeding/doublet_finding.hpp>
#include <seeding/seed_filtering.hpp>
#include <seeding/triplet_finding.hpp>

#include "utils/algorithm.hpp"

namespace traccc {

/// Seed finding
struct seed_finding : public algorithm<host_seed_container(const sp_grid&)> {

    /// Constructor for the seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param isp_container is internal spacepoint container
    seed_finding(const seedfinder_config& config)
        : m_doublet_finding(config), m_triplet_finding(config) {}

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(const sp_grid& i) const override {
        output_type seeds;

        // Run the algorithm
        seeds = {host_seed_container::header_vector(1, 0),
                 host_seed_container::item_vector(1)};

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

                host_triplet_collection triplets_per_spM;

                // triplet search from the combinations of two doublets which
                // share middle spacepoint
                for (unsigned int k = 0; k < mid_bot.first.size(); ++k) {
                    auto& doublet_mb = mid_bot.first[k];
                    auto& lb = mid_bot.second[k];

                    host_triplet_collection triplets = m_triplet_finding(
                        g2, doublet_mb, lb, mid_top.first, mid_top.second);

                    triplets_per_spM.insert(std::end(triplets_per_spM),
                                            triplets.begin(), triplets.end());
                }

                // seed filtering
                std::pair<host_triplet_collection&, host_seed_container&>
                    filter_output(triplets_per_spM, seeds);
                m_seed_filtering(g2, filter_output);
            }
        }

        return seeds;
    }

    std::vector<multiplet_statistics> get_multiplet_stats() {
        return m_multiplet_stats;
    }

    seed_statistics get_seed_stats() { return m_seed_stats; }

    private:
    doublet_finding m_doublet_finding;
    triplet_finding m_triplet_finding;
    seed_filtering m_seed_filtering;

    // for statistics pre-estimation
    seed_statistics m_seed_stats{0, 0};
    std::vector<multiplet_statistics> m_multiplet_stats;
};
}  // namespace traccc
