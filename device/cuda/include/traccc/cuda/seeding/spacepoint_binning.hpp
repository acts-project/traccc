/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/seeding/counting_grid_capacities.hpp"
#include "traccc/cuda/seeding/populating_grid.hpp"
#include "traccc/seeding/spacepoint_binning_helper.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>

// Thrust include(s).
//#include <thrust/sort.h>

namespace traccc {
namespace cuda {

/// Spacepoing binning for cuda
struct spacepoint_binning
    : public algorithm<sp_grid_buffer(host_spacepoint_container&&)> {

    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr)
        : m_config(config), m_grid_config(grid_config), m_mr(mr) {
        m_axes = get_axes(grid_config, mr);
    }

    output_type operator()(
        host_spacepoint_container&& spacepoints) const override {

        int nbins = m_axes.first.n_bins * m_axes.second.n_bins;

        // Store the container id for spacepoints
        vecmem::vector<std::pair<unsigned int, unsigned int>>
            sp_container_indices(spacepoints.total_size(), &m_mr.get());

        std::size_t k = 0;
        for (std::size_t i = 0; i < spacepoints.size(); i++) {
            std::size_t n_items = spacepoints.get_items()[i].size();

            for (std::size_t j = 0; j < n_items; j++) {
                sp_container_indices[k++] = std::make_pair(i, j);
            }
        }

        // Capacity for the bins of grid buffer
        vecmem::vector<unsigned int> grid_capacities(nbins, 0, &m_mr.get());

        // Run counting grid capacities
        traccc::cuda::counting_grid_capacities(
            m_config, m_axes.first, m_axes.second, spacepoints,
            sp_container_indices, grid_capacities, m_mr.get());

        // Create size and capacity vector for grid buffer
        std::vector<std::size_t> sizes(nbins, 0);
        std::vector<std::size_t> capacities;
        capacities.reserve(nbins);
        for (const auto& c : grid_capacities) {
            /// Note: Need to investigate why populating_grid fails without this
            /// when the data size is small.
            if (c < 60) {
                capacities.push_back(60);
                continue;
            }
            capacities.push_back(c);
        }

        // Create grid buffer
        output_type g2_buffer(m_axes.first, m_axes.second, sizes, capacities,
                              m_mr.get());

        // Run populating grid
        traccc::cuda::populating_grid(m_config, g2_buffer, spacepoints,
                                      sp_container_indices, grid_capacities,
                                      m_mr.get());

        return g2_buffer;
    }

    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::pair<output_type::axis_p0_type, output_type::axis_p1_type> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace cuda
}  // namespace traccc
