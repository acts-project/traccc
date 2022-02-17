/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// SYCL library include(s).
#include "traccc/sycl/seeding/spacepoint_binning.hpp"

#include "counting_grid_capacities.hpp"
#include "populating_grid.hpp"

// Project include(s).
#include "traccc/seeding/spacepoint_binning_helper.hpp"

namespace traccc::sycl {

spacepoint_binning::spacepoint_binning(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    vecmem::memory_resource& mr, const queue_wrapper& queue)
    : m_config(config), m_grid_config(grid_config), m_mr(mr), m_queue(queue) {
    m_axes = get_axes(grid_config, mr);
}

unsigned int spacepoint_binning::nbins() const {
    return static_cast<unsigned int>(m_axes.first.bins() *
                                     m_axes.second.bins());
}

spacepoint_binning::output_type spacepoint_binning::operator()(
    const host_spacepoint_container& spacepoints) const {

    // output object for grid of internal spacepoint
    output_type g2(m_axes.first, m_axes.second, m_mr.get());

    // capacity for the bins of grid buffer
    vecmem::vector<unsigned int> grid_capacities(g2.nbins(), 0, &m_mr.get());

    // store the container id for spacepoints
    vecmem::vector<std::pair<unsigned int, unsigned int>> sp_container_indices(
        spacepoints.total_size(), &m_mr.get());

    for (unsigned int i = 0, k = 0; i < spacepoints.size(); ++i) {
        for (unsigned int j = 0; j < spacepoints.get_items()[i].size();
             ++j, ++k) {
            sp_container_indices[k] = std::make_pair(i, j);
        }
    }

    // count the grid capacities
    traccc::sycl::counting_grid_capacities(
        m_config, g2, const_cast<host_spacepoint_container&>(spacepoints),
        sp_container_indices, grid_capacities, m_mr.get(), m_queue);

    // populate the internal spacepoints into the grid
    traccc::sycl::populating_grid(
        m_config, g2, const_cast<host_spacepoint_container&>(spacepoints),
        sp_container_indices, grid_capacities, m_mr.get(), m_queue);

    return g2;
}

}  // namespace traccc::sycl
