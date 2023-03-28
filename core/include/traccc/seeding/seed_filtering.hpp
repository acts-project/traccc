/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

namespace traccc {

/// Seed filtering to filter out the bad triplets
class seed_filtering {

    public:
    /// Constructor with the seed filter configuration
    seed_filtering(const seedfilter_config& config);

    /// Callable operator for the seed filtering
    ///
    /// @param isp_collection is internal spacepoint collection
    /// @param triplets is the vector of triplets per middle spacepoint
    ///
    /// void interface
    ///
    /// @return seeds are the vector of seeds where the new compatible seeds are
    /// added
    void operator()(const spacepoint_collection_types::host& sp_collection,
                    const sp_grid& g2, triplet_collection_types::host& triplets,
                    seed_collection_types::host& seeds) const;

    private:
    /// Seed filter configuration
    seedfilter_config m_filter_config;

};  // class seed_filtering

}  // namespace traccc
