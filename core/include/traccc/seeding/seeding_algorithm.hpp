/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/seed_finding.hpp"
#include "traccc/seeding/spacepoint_binning.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc {

/// Main algorithm for performing the track seeding on the CPU
class seeding_algorithm : public algorithm<seed_collection_types::host(
                              const spacepoint_container_types::host&)> {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr The memory resource to use
    ///
    seeding_algorithm(vecmem::memory_resource& mr);

    /// Operator executing the algorithm.
    ///
    /// @param spacepoint All spacepoints in the event
    /// @return The track seeds reconstructed from the spacepoints
    ///
    output_type operator()(
        const spacepoint_container_types::host& spacepoints) const override;

    private:
    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning m_spacepoint_binning;
    /// Sub-algorithm performing the seed finding
    seed_finding m_seed_finding;

};  // class seeding_algorithm

}  // namespace traccc
