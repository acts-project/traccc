/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::cuda::details {

/// Seed finding for cuda
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class seed_finding : public messaging {

    public:
    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param filter_config is seed filter configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    seed_finding(
        const seedfinder_config& config, const seedfilter_config& filter_config,
        const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
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

    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;

    /// Warp size of the GPU being used
    unsigned int m_warp_size;

};  // class seed_finding

}  // namespace traccc::cuda::details
