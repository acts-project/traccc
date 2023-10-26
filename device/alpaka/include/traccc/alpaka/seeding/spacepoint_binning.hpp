/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/alpaka/utils/definitions.hpp"

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// System include(s).
#include <memory>
#include <utility>

namespace traccc::alpaka {

/// Spacepoing binning executed on an Alpaka accelerator
class spacepoint_binning
    : public algorithm<sp_grid_buffer(
          const spacepoint_collection_types::const_view&)> {

    public:
    /// Constructor for the algorithm
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       const traccc::memory_resource& mr, vecmem::copy& copy);

    /// Function executing the algorithm with a view of spacepoints
    output_type operator()(const spacepoint_collection_types::const_view&
                               spacepoints_view) const override;

    private:
    /// Member variables
    seedfinder_config m_config;
    std::pair<sp_grid::axis_p0_type, sp_grid::axis_p1_type> m_axes;
    traccc::memory_resource m_mr;
    vecmem::copy& m_copy;

};  // class spacepoint_binning

}  // namespace traccc::alpaka
