/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>
#include <utility>

namespace traccc::cuda {

/// Spacepoing binning executed on a CUDA device
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class spacepoint_binning
    : public algorithm<sp_grid_buffer(
          const spacepoint_collection_types::const_view&)> {

    public:
    /// Constructor for the algorithm
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       const traccc::memory_resource& mr, vecmem::copy& copy,
                       stream& str);

    /// Function executing the algorithm with a a view of spacepoints
    sp_grid_buffer operator()(const spacepoint_collection_types::const_view&
                                  spacepoints_view) const override;

    private:
    /// Member variables
    seedfinder_config m_config;
    std::pair<sp_grid::axis_p0_type, sp_grid::axis_p1_type> m_axes;
    traccc::memory_resource m_mr;

    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;

};  // class spacepoint_binning

}  // namespace traccc::cuda
