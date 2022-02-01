/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/seeding/populating_grid.hpp"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc {
namespace cuda {

__global__ void populating_grid_kernel(
    const seedfinder_config config, sp_grid_view grid_view,
    vecmem::data::vector_view<unsigned int> grid_sizes_view,
    spacepoint_container_view spacepoints_view,
    vecmem::data::vector_view<std::pair<unsigned int, unsigned int>>
        sp_container_indices_view);

void populating_grid(
    const seedfinder_config config, sp_grid& grid,
    host_spacepoint_container& spacepoints,
    vecmem::vector<std::pair<unsigned int, unsigned int>>& sp_container_indices,
    vecmem::vector<unsigned int>& grid_capacities,
    vecmem::memory_resource& resource) {

    /// resizing the grid bin based on the capacities
    for (unsigned int i = 0; i < grid_capacities.size(); i++) {
        grid.data()[i].resize(grid_capacities[i]);
    }

    /// vector to count the number of internal spacepoints in the bin
    vecmem::vector<unsigned int> grid_sizes(grid.nbins(), 0, &resource);

    auto grid_view = get_data(grid, resource);
    auto grid_sizes_view = vecmem::get_data(grid_sizes);
    auto spacepoints_view = get_data(spacepoints, &resource);
    auto sp_container_indices_view = vecmem::get_data(sp_container_indices);

    // number of threads is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 4;
    unsigned int num_blocks = spacepoints.total_size() / num_threads + 1;

    // run the kernel
    populating_grid_kernel<<<num_blocks, num_threads>>>(
        config, grid_view, grid_sizes_view, spacepoints_view,
        sp_container_indices_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void populating_grid_kernel(
    const seedfinder_config config, sp_grid_view grid_view,
    vecmem::data::vector_view<unsigned int> grid_sizes_view,
    spacepoint_container_view spacepoints_view,
    vecmem::data::vector_view<std::pair<unsigned int, unsigned int>>
        sp_container_indices_view) {

    // Get device container for input parameters
    sp_grid_device g2_device(grid_view);
    device_spacepoint_container spacepoints_device(
        {spacepoints_view.headers, spacepoints_view.items});
    vecmem::device_vector<unsigned int> grid_sizes(grid_sizes_view);
    vecmem::device_vector<std::pair<unsigned int, unsigned int>>
        sp_container_indices(sp_container_indices_view);

    auto gid = threadIdx.x + blockIdx.x * blockDim.x;

    // kill the process before overflow
    if (gid >= sp_container_indices.size()) {
        return;
    }

    const auto& header_idx = sp_container_indices[gid].first;
    const auto& sp_idx = sp_container_indices[gid].second;

    auto spacepoints_per_module = spacepoints_device.get_items().at(header_idx);
    const auto& sp = spacepoints_per_module[sp_idx];

    /// Check out if the spacepoints can be used for seeding
    size_t r_index = is_valid_sp(config, sp);

    /// Get axis information from grid
    const auto& phi_axis = g2_device.axis_p0();
    const auto& z_axis = g2_device.axis_p1();

    /// Ignore is radius index is invalid value
    if (r_index != detray::invalid_value<size_t>()) {

        auto isp = internal_spacepoint<spacepoint>(
            spacepoints_device, {header_idx, sp_idx}, config.beamPos);

        /// Get bin index in grid
        size_t bin_index =
            phi_axis.bin(isp.phi()) + phi_axis.bins() * z_axis.bin(isp.z());

        /// increase the size for the grid bin
        auto pos = atomicAdd(&grid_sizes[bin_index], 1);

        /// replace the value
        g2_device.bin(bin_index)[pos] = std::move(isp);
    }
}

}  // namespace cuda
}  // namespace traccc
