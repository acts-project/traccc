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
struct seed_finding
    : public algorithm<const host_internal_spacepoint_container&,
                       host_seed_container> {
    /// Constructor for the seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param isp_container is internal spacepoint container
    seed_finding(seedfinder_config& config)
        : m_doublet_finding(config), m_triplet_finding(config) {}

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(const input_type& i) const override {
        output_type result;
        this->operator()(i, result);
        return result;
    }

    /// Callable operator for the seed finding
    ///
    /// void interface
    ///
    /// @return seed_collection is the vector of seeds per event
    void operator()(const input_type& i, output_type& o) const {
        // input
        const auto& isp_container = i;
        // output
        auto& seeds = o;

        // Run the algorithm
        seeds = {host_seed_container::header_vector(1, 0),
                 host_seed_container::item_vector(1)};

        const bool bottom = true;
        const bool top = false;

        // iterate over grid bins
        for (unsigned int i = 0; i < isp_container.size(); ++i) {
            auto& bin_information = isp_container.get_headers()[i];
            auto& spM_collection = isp_container.get_items()[i];

            // multiplet statistics for GPU vector size estimation
            /*
            multiplet_statistics stats({0, 0, 0, 0});
            stats.n_spM = spM_collection.size();
            */

            /// iterate over middle spacepoints
            for (unsigned int j = 0; j < spM_collection.size(); ++j) {
                sp_location spM_location({i, j});

                // middule-bottom doublet search
                auto mid_bot = m_doublet_finding(
                    {isp_container, bin_information, spM_location, bottom});

                if (mid_bot.first.empty())
                    continue;

                // middule-top doublet search
                auto mid_top = m_doublet_finding(
                    {isp_container, bin_information, spM_location, top});

                if (mid_top.first.empty())
                    continue;

                host_triplet_collection triplets_per_spM;

                // triplet search from the combinations of two doublets which
                // share middle spacepoint
                for (unsigned int k = 0; k < mid_bot.first.size(); ++k) {
                    auto& doublet_mb = mid_bot.first[k];
                    auto& lb = mid_bot.second[k];

                    host_triplet_collection triplets =
                        m_triplet_finding({isp_container, doublet_mb, lb,
                                           mid_top.first, mid_top.second});

                    triplets_per_spM.insert(std::end(triplets_per_spM),
                                            triplets.begin(), triplets.end());
                }

                // seed filtering
                std::pair<host_triplet_collection&, host_seed_container&>
                    filter_output(triplets_per_spM, seeds);
                m_seed_filtering(isp_container, filter_output);

                /*
                stats.n_mid_bot_doublets += mid_bot.first.size();
                stats.n_mid_top_doublets += mid_top.first.size();
                stats.n_triplets += triplets_per_spM.size();
                */
            }

            /*
            m_multiplet_stats.push_back(stats);
            */
        }

        /*
        m_seed_stats = seed_statistics({0, 0});
        for (size_t i = 0; i < isp_container.headers.size(); ++i) {
            m_seed_stats.n_internal_sp += isp_container.items[i].size();
        }

        m_seed_stats.n_seeds = seeds.items[0].size();
        */
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
    seed_statistics m_seed_stats;
    std::vector<multiplet_statistics> m_multiplet_stats;
};
}  // namespace traccc
