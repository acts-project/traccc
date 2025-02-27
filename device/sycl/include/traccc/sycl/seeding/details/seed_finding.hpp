/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::sycl::details {

// Sycl seeding function object
class seed_finding : public messaging {

    public:
    /// Constructor for the sycl seed finding
    ///
    /// @param config   is seed finder configuration parameters
    /// @param filter_config is seed filter configuration parameters
    /// @param mr       is a struct of memory resources (shared or
    /// host & device)
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue    is a wrapper for the sycl queue for kernel
    /// invocation
    seed_finding(
        const seedfinder_config& config, const seedfilter_config& filter_config,
        const traccc::memory_resource& mr, vecmem::copy& copy,
        queue_wrapper queue,
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
    /// Private member variables
    seedfinder_config m_seedfinder_config;
    seedfilter_config m_seedfilter_config;
    traccc::memory_resource m_mr;
    mutable queue_wrapper m_queue;
    vecmem::copy& m_copy;

};  // class seed_finding

}  // namespace traccc::sycl::details
