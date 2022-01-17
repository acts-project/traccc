/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <CL/sycl.hpp>

#include "sycl/seeding/counting_grid_capacities.hpp"
#include <sycl/seeding/populating_grid.hpp>
#include <seeding/spacepoint_binning_helper.hpp>
#include <utils/algorithm.hpp>

#include "vecmem/memory/sycl/device_memory_resource.hpp"

namespace traccc {
namespace sycl {

/// Spacepoing binning for cuda
struct spacepoint_binning
    : public algorithm<sp_grid(host_spacepoint_container&&)> {

    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr,
                       ::sycl::queue* q)
        : m_config(config), m_grid_config(grid_config), m_mr(mr), m_q(q) {
        m_axes = get_axes(grid_config);
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
        traccc::sycl::counting_grid_capacities(m_config, g2, spacepoints,
                                               sp_container_indices,
                                               grid_capacities, m_mr.get(), m_q);

        // populate the internal spacepoints into the grid
        traccc::sycl::populating_grid(m_config, g2, spacepoints,
                                      sp_container_indices, grid_capacities,
                                      m_mr.get(), m_q);

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
    std::pair<output_type::axis_p0_t, output_type::axis_p1_t> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    ::sycl::queue* m_q;
};

}  // namespace sycl
}  // namespace traccc