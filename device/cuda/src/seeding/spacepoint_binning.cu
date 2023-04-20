/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/spacepoint_binning.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s).
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
    spacepoint_collection_types::const_view spacepoints,
    vecmem::data::vector_view<unsigned int> grid_capacities) {

    device::count_grid_capacities(threadIdx.x + blockIdx.x * blockDim.x, config,
                                  phi_axis, z_axis, spacepoints,
                                  grid_capacities);
}

/// CUDA kernel for running @c traccc::device::populate_grid
__global__ void populate_grid(
    seedfinder_config config,
    spacepoint_collection_types::const_view spacepoints, sp_grid_view grid) {

    device::populate_grid(threadIdx.x + blockIdx.x * blockDim.x, config,
                          spacepoints, grid);
}

}  // namespace kernels

spacepoint_binning::spacepoint_binning(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_config(config.toInternalUnits()),
      m_axes(get_axes(grid_config.toInternalUnits(),
                      (mr.host ? *(mr.host) : mr.main))),
      m_mr(mr),
      m_copy(copy),
      m_stream(str) {}

sp_grid_buffer spacepoint_binning::operator()(
    const spacepoint_collection_types::const_view& spacepoints_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the spacepoint sizes from the view
    auto sp_size = m_copy.get_size(spacepoints_view);

    // Set up the container that will be filled with the required capacities for
    // the spacepoint grid.
    const std::size_t grid_bins = m_axes.first.n_bins * m_axes.second.n_bins;
    vecmem::data::vector_buffer<unsigned int> grid_capacities_buff(grid_bins,
                                                                   m_mr.main);
    m_copy.setup(grid_capacities_buff);
    m_copy.memset(grid_capacities_buff, 0);
    vecmem::data::vector_view<unsigned int> grid_capacities_view =
        grid_capacities_buff;

    // Calculate the number of threads and thread blocks to run the kernels for.
    const unsigned int num_threads = WARP_SIZE * 8;
    const unsigned int num_blocks = (sp_size + num_threads - 1) / num_threads;

    // Fill the grid capacity container.
    kernels::count_grid_capacities<<<num_blocks, num_threads, 0, stream>>>(
        m_config, m_axes.first, m_axes.second, spacepoints_view,
        grid_capacities_view);
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy grid capacities back to the host
    vecmem::vector<unsigned int> grid_capacities_host(m_mr.host ? m_mr.host
                                                                : &(m_mr.main));
    m_copy(grid_capacities_buff, grid_capacities_host);

    m_stream.synchronize();
    // Create the grid buffer.
    sp_grid_buffer grid_buffer(
        m_axes.first, m_axes.second,
        std::vector<std::size_t>(grid_capacities_host.begin(),
                                 grid_capacities_host.end()),
        m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable);
    m_copy.setup(grid_buffer._buffer);
    sp_grid_view grid_view = grid_buffer;

    // Populate the grid.
    kernels::populate_grid<<<num_blocks, num_threads, 0, stream>>>(
        m_config, spacepoints_view, grid_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    m_stream.synchronize();

    // Return the freshly filled buffer.
    return grid_buffer;
}

}  // namespace traccc::cuda
