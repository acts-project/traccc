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
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

// Sycl seeding function object
class seed_finding
    : public algorithm<vecmem::data::vector_buffer<seed>(const spacepoint_container_const_view&,
                                            const sp_grid_const_view&)> {

    public:
    /// Constructor for the sycl seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param stats_config experiment-dependent statistics estimator
    /// @param mr vecmem memory resource
    /// @param q sycl queue for kernel scheduling
    seed_finding(const seedfinder_config& config, vecmem::memory_resource& mr,
                 queue_wrapper queue);

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(const spacepoint_container_const_view& spacepoints_view,
                           const sp_grid_const_view& g2_view) const override;

    private:
    seedfinder_config m_seedfinder_config;
    seedfilter_config m_seedfilter_config;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl
