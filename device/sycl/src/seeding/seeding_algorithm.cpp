/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/sycl/seeding/seeding_algorithm.hpp"

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <cmath>

namespace traccc::sycl {

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     const traccc::memory_resource& mr,
                                     vecmem::copy& copy,
                                     const queue_wrapper& queue)
    : m_spacepoint_binning(finder_config, grid_config, mr, copy, queue),
      m_seed_finding(finder_config, filter_config, mr, copy, queue) {}

seeding_algorithm::output_type seeding_algorithm::operator()(
    const spacepoint_collection_types::const_view& spacepoints_view) const {

    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning binning_alg(m_finder_config, m_grid_config, m_mr, m_copy,
                                   m_queue);
    /// Sub-algorithm performing the seed finding
    seed_finding finding_alg(m_finder_config, m_filter_config, m_mr, m_copy,
                             m_queue);

    return finding_alg(spacepoints_view, binning_alg(spacepoints_view));
}

}  // namespace traccc::sycl
