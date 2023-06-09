/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/cuda/seeding/seed_finding.hpp"
#include "traccc/cuda/seeding/spacepoint_binning.hpp"
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// traccc library include(s).
#include "traccc/utils/memory_resource.hpp"

namespace traccc::cuda {

/// Main algorithm for performing the track seeding on an NVIDIA GPU
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class seeding_algorithm : public algorithm<seed_collection_types::buffer(
                              const spacepoint_collection_types::const_view&)> {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr The memory resource to use
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    ///
    seeding_algorithm(const seedfinder_config& finder_config,
                      const spacepoint_grid_config& grid_config,
                      const seedfilter_config& filter_config,
                      const traccc::memory_resource& mr, vecmem::copy& copy,
                      stream& str);

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints_view is a view of all spacepoints in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(const spacepoint_collection_types::const_view&
                               spacepoints_view) const override;

    private:
    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning m_spacepoint_binning;
    /// Sub-algorithm performing the seed finding
    seed_finding m_seed_finding;

};  // class seeding_algorithm

}  // namespace traccc::cuda
