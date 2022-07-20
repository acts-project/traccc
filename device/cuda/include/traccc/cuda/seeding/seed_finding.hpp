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
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::cuda {

/// Seed finding for cuda
class seed_finding
    : public algorithm<vecmem::data::vector_buffer<seed>(
          const spacepoint_container_types::const_view&,
          const sp_grid_const_view&)>,
      public algorithm<vecmem::data::vector_buffer<seed>(
          const spacepoint_container_types::buffer&, const sp_grid_buffer&)> {

    public:
    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param mr vecmem memory resource
    seed_finding(const seedfinder_config& config,
                 const traccc::memory_resource& mr);

    /// Callable operator for the seed finding
    ///
    /// @param spacepoints_view     is a view of all spacepoints in the event
    /// @param g2_view              is a view of the spacepoint grid
    /// @return                     a vector buffer of seeds
    ///
    vecmem::data::vector_buffer<seed> operator()(
        const spacepoint_container_types::const_view& spacepoints_view,
        const sp_grid_const_view& g2_view) const override;

    /// Callable operator for the seed finding
    ///
    /// @param spacepoints_buffer   is a buffer of all spacepoints in the event
    /// @param g2_buffer            is a buffer of the spacepoint grid
    /// @return                     a vector buffer of seeds
    ///
    vecmem::data::vector_buffer<seed> operator()(
        const spacepoint_container_types::buffer& spacepoints_buffer,
        const sp_grid_buffer& g2_buffer) const override;

    private:
    /// Implementation for the public seed finding operators.
    vecmem::data::vector_buffer<seed> operator()(
        const spacepoint_container_types::const_view& spacepoints_view,
        const sp_grid_const_view& g2_view,
        const std::vector<unsigned int>& grid_sizes) const;

    private:
    seedfinder_config m_seedfinder_config;
    seedfilter_config m_seedfilter_config;
    traccc::memory_resource m_mr;
    std::unique_ptr<vecmem::copy> m_copy;
};

}  // namespace traccc::cuda
