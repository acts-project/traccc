/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/get_size.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/details/spacepoint_binning.hpp"

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
    seedfinder_config config,
    traccc::details::spacepoint_grid_types::host::axis_p0_type phi_axis,
    traccc::details::spacepoint_grid_types::host::axis_p1_type z_axis,
    edm::spacepoint_collection::const_view spacepoints,
    vecmem::data::vector_view<unsigned int> grid_capacities) {

    device::count_grid_capacities(details::global_index1(), config, phi_axis,
                                  z_axis, spacepoints, grid_capacities);
}

/// CUDA kernel for running @c traccc::device::populate_grid
__global__ void populate_grid(
    seedfinder_config config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::view grid) {

    device::populate_grid(details::global_index1(), config, spacepoints, grid);
}

}  // namespace kernels

namespace details {
spacepoint_binning::spacepoint_binning(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(config),
      m_axes(get_axes(grid_config, (mr.host ? *(mr.host) : mr.main))),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

traccc::details::spacepoint_grid_types::buffer spacepoint_binning::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Staging area for copying sizes from device to host
    vecmem::unique_alloc_ptr<unsigned int> size_staging_ptr =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));

    // Get the spacepoint sizes from the view
    const auto sp_size =
        get_size(spacepoints_view, size_staging_ptr.get(), stream);

    if (sp_size == 0) {
        return {m_axes.first, m_axes.second, {}, m_mr.main, m_mr.host};
    }

    // Set up the container that will be filled with the required capacities for
    // the spacepoint grid.
    const unsigned int grid_bins = m_axes.first.n_bins * m_axes.second.n_bins;
    vecmem::data::vector_buffer<unsigned int> grid_capacities_buff(grid_bins,
                                                                   m_mr.main);
    m_copy.setup(grid_capacities_buff)->ignore();
    m_copy.memset(grid_capacities_buff, 0)->ignore();
    vecmem::data::vector_view<unsigned int> grid_capacities_view =
        grid_capacities_buff;

    // Calculate the number of threads and thread blocks to run the kernels for.
    const unsigned int num_threads = m_warp_size * 8;
    const unsigned int num_blocks = (sp_size + num_threads - 1) / num_threads;

    // Fill the grid capacity container.
    kernels::count_grid_capacities<<<num_blocks, num_threads, 0, stream>>>(
        m_config, m_axes.first, m_axes.second, spacepoints_view,
        grid_capacities_view);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy grid capacities back to the host
    vecmem::vector<unsigned int> grid_capacities_host(m_mr.host ? m_mr.host
                                                                : &(m_mr.main));
    m_copy(grid_capacities_buff, grid_capacities_host)->wait();

    // Create the grid buffer.
    traccc::details::spacepoint_grid_types::buffer grid_buffer(
        m_axes.first, m_axes.second,
        std::vector<std::size_t>(grid_capacities_host.begin(),
                                 grid_capacities_host.end()),
        m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable);
    m_copy.setup(grid_buffer._buffer)->ignore();

    // Populate the grid.
    kernels::populate_grid<<<num_blocks, num_threads, 0, stream>>>(
        m_config, spacepoints_view, grid_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the freshly filled buffer.
    return grid_buffer;
}

}  // namespace details
}  // namespace traccc::cuda
