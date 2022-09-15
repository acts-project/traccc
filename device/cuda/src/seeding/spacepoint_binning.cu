/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/spacepoint_binning.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s).
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/seeding/device/count_grid_capacities.hpp"
#include "traccc/seeding/device/populate_grid.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::count_grid_capacities
__global__ void count_grid_capacities(
    seedfinder_config config, sp_grid::axis_p0_type phi_axis,
    sp_grid::axis_p1_type z_axis,
    spacepoint_container_types::const_view spacepoints,
    vecmem::data::vector_view<const device::prefix_sum_element_t> sp_prefix_sum,
    vecmem::data::vector_view<unsigned int> grid_capacities) {

    device::count_grid_capacities(threadIdx.x + blockIdx.x * blockDim.x, config,
                                  phi_axis, z_axis, spacepoints, sp_prefix_sum,
                                  grid_capacities);
}

/// CUDA kernel for running @c traccc::device::populate_grid
__global__ void populate_grid(
    seedfinder_config config,
    spacepoint_container_types::const_view spacepoints,
    vecmem::data::vector_view<const device::prefix_sum_element_t> sp_prefix_sum,
    sp_grid_view grid) {

    device::populate_grid(threadIdx.x + blockIdx.x * blockDim.x, config,
                          spacepoints, sp_prefix_sum, grid);
}

}  // namespace kernels

spacepoint_binning::spacepoint_binning(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    const traccc::memory_resource& mr)
    : m_config(config.toInternalUnits()),
      m_axes(get_axes(grid_config.toInternalUnits(),
                      (mr.host ? *(mr.host) : mr.main))),
      m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

sp_grid_buffer spacepoint_binning::operator()(
    const spacepoint_container_types::const_view& spacepoints_view) const {

    // Get the spacepoint sizes from the view
    auto sp_sizes = m_copy->get_sizes(spacepoints_view.items);

    return this->operator()(spacepoints_view, sp_sizes);
}

sp_grid_buffer spacepoint_binning::operator()(
    const spacepoint_container_types::buffer& spacepoints_buffer) const {

    // Get the spacepoint sizes from the buffer
    auto sp_sizes = m_copy->get_sizes(spacepoints_buffer.items);

    return this->operator()(spacepoints_buffer, sp_sizes);
}

sp_grid_buffer spacepoint_binning::operator()(
    const spacepoint_container_types::const_view& spacepoints_view,
    const std::vector<unsigned int>& sp_sizes) const {

    // Get the prefix sum for the spacepoints using buffer.
    const device::prefix_sum_t sp_prefix_sum = device::get_prefix_sum(
        sp_sizes, (m_mr.host ? *(m_mr.host) : m_mr.main));

    // Set up the buffer of the prefix sum and its view
    vecmem::data::vector_buffer<device::prefix_sum_element_t>
        sp_prefix_sum_buff(sp_prefix_sum.size(), m_mr.main);
    m_copy->setup(sp_prefix_sum_buff);
    (*m_copy)(vecmem::get_data(sp_prefix_sum), sp_prefix_sum_buff,
              vecmem::copy::type::copy_type::host_to_device);
    vecmem::data::vector_view<device::prefix_sum_element_t> sp_prefix_sum_view =
        sp_prefix_sum_buff;

    // Set up the container that will be filled with the required capacities for
    // the spacepoint grid.
    const std::size_t grid_bins = m_axes.first.n_bins * m_axes.second.n_bins;
    vecmem::data::vector_buffer<unsigned int> grid_capacities_buff(grid_bins,
                                                                   m_mr.main);
    m_copy->setup(grid_capacities_buff);
    m_copy->memset(grid_capacities_buff, 0);
    vecmem::data::vector_view<unsigned int> grid_capacities_view =
        grid_capacities_buff;

    // Calculate the number of threads and thread blocks to run the kernels for.
    const unsigned int num_threads = WARP_SIZE * 8;
    const unsigned int num_blocks = sp_prefix_sum.size() / num_threads + 1;

    // Fill the grid capacity container.
    kernels::count_grid_capacities<<<num_blocks, num_threads>>>(
        m_config, m_axes.first, m_axes.second, spacepoints_view,
        sp_prefix_sum_view, grid_capacities_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy grid capacities back to the host
    vecmem::vector<unsigned int> grid_capacities_host(m_mr.host ? m_mr.host
                                                                : &(m_mr.main));
    (*m_copy)(grid_capacities_buff, grid_capacities_host);

    // Create the grid buffer.
    sp_grid_buffer grid_buffer(
        m_axes.first, m_axes.second, std::vector<std::size_t>(grid_bins, 0),
        std::vector<std::size_t>(grid_capacities_host.begin(),
                                 grid_capacities_host.end()),
        m_mr.main, m_mr.host);
    m_copy->setup(grid_buffer._buffer);
    sp_grid_view grid_view = grid_buffer;

    // Populate the grid.
    kernels::populate_grid<<<num_blocks, num_threads>>>(
        m_config, spacepoints_view, sp_prefix_sum_view, grid_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Return the freshly filled buffer.
    return grid_buffer;
}

}  // namespace traccc::cuda
