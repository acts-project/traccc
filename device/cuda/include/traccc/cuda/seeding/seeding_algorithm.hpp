/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/cuda/seeding/seed_finding.hpp"
#include "traccc/cuda/seeding/spacepoint_binning.hpp"

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// traccc library include(s).
#include "traccc/utils/memory_resource.hpp"

namespace traccc::cuda {

/// Main algorithm for performing the track seeding on an NVIDIA GPU
class seeding_algorithm : public algorithm<seed_collection_types::buffer(
                              const spacepoint_container_types::const_view&)> {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr The memory resource to use
    ///
    seeding_algorithm(const traccc::memory_resource& mr);

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints_view is a view of all spacepoints in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    seed_collection_types::buffer operator()(
        const spacepoint_container_types::const_view& spacepoints_view)
        const override;

    private:
    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning m_spacepoint_binning;
    /// Sub-algorithm performing the seed finding
    seed_finding m_seed_finding;

};  // class seeding_algorithm

}  // namespace traccc::cuda
