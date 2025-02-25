/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/sycl/seeding/details/seed_finding.hpp"
#include "traccc/sycl/seeding/details/spacepoint_binning.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::sycl {

/// Main algorithm for performing the track seeding using oneAPI/SYCL
class seeding_algorithm : public algorithm<edm::seed_collection::buffer(
                              const edm::spacepoint_collection::const_view&)>,
                          public messaging {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr is a struct of memory resources (shared or host & device)
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue The SYCL queue to work with
    ///
    seeding_algorithm(
        const seedfinder_config& finder_config,
        const spacepoint_grid_config& grid_config,
        const seedfilter_config& filter_config,
        const traccc::memory_resource& mr, vecmem::copy& copy,
        const queue_wrapper& queue,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints_view is a view of all spacepoints in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(const edm::spacepoint_collection::const_view&
                               spacepoints_view) const override;

    private:
    /// Tool performing the spacepoint binning
    details::spacepoint_binning m_binning;
    /// Tool performing the seed finding
    details::seed_finding m_finding;

};  // class seeding_algorithm

}  // namespace traccc::sycl
