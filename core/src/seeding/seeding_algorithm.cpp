/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/seeding_algorithm.hpp"

#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <cmath>
#include <iostream>

namespace traccc {

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     vecmem::memory_resource& mr)
    : m_spacepoint_binning(finder_config, grid_config, mr),
      m_seed_finding(finder_config, filter_config) {}

seeding_algorithm::output_type seeding_algorithm::operator()(
    const spacepoint_collection_types::host& spacepoints) const {

    return m_seed_finding(spacepoints, m_spacepoint_binning(spacepoints));
}

}  // namespace traccc
