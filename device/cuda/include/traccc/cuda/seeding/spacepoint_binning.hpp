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
    : public algorithm<sp_grid(host_spacepoint_container&&)> {

    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr)
        : m_config(config), m_grid_config(grid_config), m_mr(mr) {
        m_axes = get_axes(grid_config, mr);
    }

    unsigned int nbins() const {
        return static_cast<unsigned int>(m_axes.first.bins() *
                                         m_axes.second.bins());
    }

    output_type operator()(
        host_spacepoint_container&& spacepoints) const override {

        // output object for grid of internal spacepoint
        output_type g2(m_axes.first, m_axes.second, m_mr.get());

        // capacity for the bins of grid buffer
        vecmem::vector<unsigned int> grid_capacities(g2.nbins(), 0,
                                                     &m_mr.get());

        // store the container id for spacepoints
        vecmem::vector<std::pair<unsigned int, unsigned int>>
            sp_container_indices(spacepoints.total_size(), &m_mr.get());

        int k = 0;
        for (unsigned int i = 0; i < spacepoints.size(); i++) {
            for (unsigned int j = 0; j < spacepoints.get_items()[i].size();
                 j++) {
                sp_container_indices[k++] = std::make_pair(i, j);
            }
        }

        // count the grid capacities
        traccc::cuda::counting_grid_capacities(m_config, g2, spacepoints,
                                               sp_container_indices,
                                               grid_capacities, m_mr.get());

        // populate the internal spacepoints into the grid
        traccc::cuda::populating_grid(m_config, g2, spacepoints,
                                      sp_container_indices, grid_capacities,
                                      m_mr.get());

        /// It is OPTIONAL to do sorting with the radius of spacepoint,
        /// since the sorting barely impacts seed matching ratio between cpu and
        /// cuda
        /*
        for (unsigned int i = 0; i < g2.nbins(); i++){
            auto& g2_bin = g2.data()[i];

            // cpu sort
            //std::sort(g2_bin.begin(), g2_bin.end());

            // thrust sort
            thrust::sort(g2_bin.begin(),g2_bin.end());
        }
        */

        return g2;
    }

    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::pair<output_type::axis_p0_type, output_type::axis_p1_type> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace cuda
}  // namespace traccc
