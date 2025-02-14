/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <utility>

namespace traccc::alpaka::details {

/// Spacepoing binning executed on an Alpaka accelerator
class spacepoint_binning : public messaging {

    public:
    /// Constructor for the algorithm
    spacepoint_binning(
        const seedfinder_config& config,
        const spacepoint_grid_config& grid_config,
        const traccc::memory_resource& mr, vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Function executing the algorithm with a view of spacepoints
    traccc::details::spacepoint_grid_types::buffer operator()(
        const edm::spacepoint_collection::const_view& spacepoints) const;

    private:
    /// Member variables
    seedfinder_config m_config;
    std::pair<traccc::details::spacepoint_grid_types::host::axis_p0_type,
              traccc::details::spacepoint_grid_types::host::axis_p1_type>
        m_axes;
    traccc::memory_resource m_mr;
    vecmem::copy& m_copy;
};  // class spacepoint_binning

}  // namespace traccc::alpaka::details
