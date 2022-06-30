/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::cuda {

/// Seed finding for cuda
class seed_finding
    : public algorithm<host_seed_collection(
          const spacepoint_container_types::view&, const sp_grid_const_view&)> {

    public:
    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param mr vecmem memory resource
    seed_finding(const seedfinder_config& config, vecmem::memory_resource& mr);

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(const spacepoint_container_types::view& spacepoints,
                           const sp_grid_const_view& g2_view) const override;

    private:
    seedfinder_config m_seedfinder_config;
    seedfilter_config m_seedfilter_config;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::cuda
