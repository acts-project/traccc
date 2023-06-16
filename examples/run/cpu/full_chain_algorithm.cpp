/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

namespace traccc {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& mr, unsigned int,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config)
    : m_clusterization(mr),
      m_spacepoint_formation(mr),
      m_seeding(finder_config, grid_config, filter_config, mr),
      m_track_parameter_estimation(mr),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config) {}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    const spacepoint_formation::output_type spacepoints =
        m_spacepoint_formation(m_clusterization(cells, modules), modules);

    return m_track_parameter_estimation(spacepoints, m_seeding(spacepoints),
                                        {0.f, 0.f, m_finder_config.bFieldInZ});
}

}  // namespace traccc
