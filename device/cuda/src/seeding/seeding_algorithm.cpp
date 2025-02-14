/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/cuda/seeding/seeding_algorithm.hpp"

namespace traccc::cuda {

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     const traccc::memory_resource& mr,
                                     vecmem::copy& copy, stream& str,
                                     std::unique_ptr<const Logger> logger)
    : messaging(logger->clone()),
      m_binning(finder_config, grid_config, mr, copy, str,
                logger->cloneWithSuffix("BinningAlg")),
      m_finding(finder_config, filter_config, mr, copy, str,
                logger->cloneWithSuffix("SeedFindingAlg")) {}

seeding_algorithm::output_type seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view) const {

    return m_finding(spacepoints_view, m_binning(spacepoints_view));
}

}  // namespace traccc::cuda
