/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"

// System include(s).
#include <functional>

namespace traccc {

/// spacepoint binning
class spacepoint_binning
    : public algorithm<sp_grid(const spacepoint_collection_types::host&)> {

    public:
    /// Constructor for the spacepoint binning
    ///
    /// @param config is seed finder configuration parameters
    /// @param grid_config is for spacepoint grid parameter
    /// @param mr is the vecmem memory resource
    ///
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr);

    /// Operator executing the algorithm
    ///
    /// @param sp_collection All of the spacepoints of the event
    /// @return The spacepoints arranged in a Phi-Z grid
    ///
    output_type operator()(
        const spacepoint_collection_types::host& sp_collection) const override;

    private:
    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::pair<output_type::axis_p0_type, output_type::axis_p1_type> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
