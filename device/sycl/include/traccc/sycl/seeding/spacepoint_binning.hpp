/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>
#include <memory>
#include <utility>

namespace traccc::sycl {

/// Spacepoing binning executed on a SYCL device
class spacepoint_binning : public algorithm<sp_grid_buffer(
                               const spacepoint_container_types::const_view&)> {

    public:
    /// Constructor for the algorithm
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       const traccc::memory_resource& mr, queue_wrapper queue);

    /// Function executing the algorithm with a a view of spacepoints
    sp_grid_buffer operator()(const spacepoint_container_types::const_view&
                                  spacepoints_view) const override;

    private:
    /// Member variables
    seedfinder_config m_config;
    std::pair<sp_grid::axis_p0_type, sp_grid::axis_p1_type> m_axes;
    traccc::memory_resource m_mr;
    mutable queue_wrapper m_queue;
    std::unique_ptr<vecmem::copy> m_copy;

};  // class spacepoint_binning

}  // namespace traccc::sycl
