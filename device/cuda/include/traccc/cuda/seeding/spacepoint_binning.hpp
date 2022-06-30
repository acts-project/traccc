/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>
#include <utility>

namespace traccc::cuda {

/// Spacepoing binning executed on a CUDA device
class spacepoint_binning : public algorithm<sp_grid_buffer(
                               const spacepoint_container_types::view&)> {

    public:
    /// Constructor for the algorithm
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr);

    /// Function executing the algorithm
    sp_grid_buffer operator()(
        const spacepoint_container_types::view& sp_data) const override;

    private:
    seedfinder_config m_config;
    std::pair<sp_grid::axis_p0_type, sp_grid::axis_p1_type> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class spacepoint_binning

}  // namespace traccc::cuda
