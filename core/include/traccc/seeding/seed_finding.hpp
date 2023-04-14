/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/doublet_finding.hpp"
#include "traccc/seeding/seed_filtering.hpp"
#include "traccc/seeding/triplet_finding.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Seed finding
class seed_finding
    : public algorithm<seed_collection_types::host(
          const spacepoint_collection_types::host&, const sp_grid&)> {

    public:
    /// Constructor for the seed finding
    ///
    /// @param find_config is seed finder configuration parameters
    /// @param filter_config is the seed filter configuration
    ///
    seed_finding(const seedfinder_config& find_config,
                 const seedfilter_config& filter_config);

    /// Callable operator for the seed finding
    ///
    /// @param sp_collection All spacepoints in the event
    /// @param g2 The same spacepoints arranged in a 2D Phi-Z grid
    /// @return seed_collection is the vector of seeds per event
    ///
    output_type operator()(
        const spacepoint_collection_types::host& sp_collection,
        const sp_grid& g2) const override;

    private:
    /// Algorithm performing the mid bottom doublet finding
    doublet_finding<details::spacepoint_type::bottom> m_midBot_finding;
    /// Algorithm performing the mid top doublet finding
    doublet_finding<details::spacepoint_type::top> m_midTop_finding;
    /// Algorithm performing the triplet finding
    triplet_finding m_triplet_finding;
    /// Algorithm performing the seed selection
    seed_filtering m_seed_filtering;

};  // class seed_finding

}  // namespace traccc
