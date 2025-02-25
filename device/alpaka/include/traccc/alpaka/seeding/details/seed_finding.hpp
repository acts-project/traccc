/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::alpaka::details {

/// Seed finding for alpaka
class seed_finding : public messaging {

    public:
    /// Constructor for the alpaka seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param filter_config is seed filter configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param mr vecmem memory resource
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    seed_finding(
        const seedfinder_config& config, const seedfilter_config& filter_config,
        const traccc::memory_resource& mr, vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator for the seed finding
    ///
    /// @param spacepoints_view     is a view of all spacepoints in the event
    /// @param g2_view              is a view of the spacepoint grid
    /// @return                     a vector buffer of seeds
    ///
    edm::seed_collection::buffer operator()(
        const edm::spacepoint_collection::const_view& spacepoints_view,
        const traccc::details::spacepoint_grid_types::const_view& g2_view)
        const;

    private:
    seedfinder_config m_seedfinder_config;
    seedfilter_config m_seedfilter_config;
    traccc::memory_resource m_mr;
    vecmem::copy& m_copy;

};  // class seed_finding

}  // namespace traccc::alpaka::details
