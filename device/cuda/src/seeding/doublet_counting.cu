/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/seeding/doublet_counting.hpp"
#include "traccc/cuda/utils/cuda_helper.cuh"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc {
namespace cuda {

__global__ void set_zero_kernel(doublet_counter_container_view dcc_view) {
    device_doublet_counter_container dcc_device(
        {dcc_view.headers, dcc_view.items});
    dcc_device.get_headers().at(threadIdx.x).zeros();
}

/// Forward declaration of doublet counting kernel
/// The number of mid-bot and mid-top doublets are counted for all spacepoints
/// and recorded into doublet counter container if the number of doublets are
/// larger than zero
///
/// @param config seed finder config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_count_view vecmem container for doublet_counter
__global__ void doublet_counting_kernel(
    const seedfinder_config config, sp_grid_view internal_sp_view,
    doublet_counter_container_view doublet_count_view);

void doublet_counting(const seedfinder_config& config,
                      sp_grid_view internal_sp_view,
                      doublet_counter_container_view dcc_view,
                      vecmem::memory_resource& resource) {

    unsigned int nbins = internal_sp_view._data_view.m_size;

    // zero initialization
    set_zero_kernel<<<1, nbins>>>(dcc_view);
    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // The thread-block is desinged to make each thread count the number of
    // doublets per middle spacepoint

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as
    // num_middle_sp_per_bin / num_threads + 1
    unsigned int num_blocks = 0;
    for (size_t i = 0; i < nbins; ++i) {
        num_blocks +=
            internal_sp_view._data_view.m_ptr[i].size() / num_threads + 1;
    }

    // run the kernel
    doublet_counting_kernel<<<num_blocks, num_threads>>>(
        config, internal_sp_view, dcc_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void doublet_counting_kernel(
    const seedfinder_config config, sp_grid_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view) {

    // Get device container for input parameters
    sp_grid_device internal_sp_device(internal_sp_view);

    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});

    // Get bin and spacepoint index
    unsigned int bin_idx(0), sp_idx(0);
    cuda_helper::find_idx_on_jagged_vector(internal_sp_device.data(), bin_idx,
                                           sp_idx);

    // get internal spacepoints for current bin
    auto internal_sp_per_bin = internal_sp_device.bin(bin_idx);

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin
    auto& num_compat_spM_per_bin =
        doublet_counter_device.get_headers().at(bin_idx).n_spM;
    auto& num_mid_bot_per_bin =
        doublet_counter_device.get_headers().at(bin_idx).n_mid_bot;
    auto& num_mid_top_per_bin =
        doublet_counter_device.get_headers().at(bin_idx).n_mid_top;

    auto doublet_counter_per_bin =
        doublet_counter_device.get_items().at(bin_idx);
    // kill the process before overflow
    if (sp_idx >= doublet_counter_per_bin.size()) {
        return;
    }

    // zero initialization for the number of doublets per thread (or middle sp)
    unsigned int n_mid_bot = 0;
    unsigned int n_mid_top = 0;

    // zero initialization for the number of doublets per bin
    doublet_counter_per_bin[sp_idx].n_mid_bot = 0;
    doublet_counter_per_bin[sp_idx].n_mid_top = 0;

    // middle spacepoint index
    auto spM_loc = sp_location({bin_idx, sp_idx});
    // middle spacepoint
    const auto& isp = internal_sp_per_bin[sp_idx];

    // get the neighbor indices
    auto phi_bins =
        internal_sp_device.axis_p0().range(isp.phi(), config.neighbor_scope);
    auto z_bins =
        internal_sp_device.axis_p1().range(isp.z(), config.neighbor_scope);
    auto i_p = phi_bins[0];
    auto i_z = z_bins[0];

    for (unsigned int c = 0; c < config.get_max_neighbor_bins(); c++) {

        auto neighbors = internal_sp_device.bin(i_p, i_z);

        for (auto& neigh_isp : neighbors) {

            // Check if middle and bottom sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     true)) {
                n_mid_bot++;
            }

            // Check if middle and top sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     false)) {
                n_mid_top++;
            }
        }

        i_z++;
        // terminate if we went through all neighbot bins
        if (i_z > z_bins[1] && i_p == phi_bins[1]) {
            break;
        }

        // increse phi_index and reset i_z
        // if i_z is larger than last neighbor index
        if (i_z > z_bins[1]) {
            i_p++;
            i_p = i_p % internal_sp_device.axis_p0().bins();
            i_z = z_bins[0];
        }
    }

    // if number of mid-bot and mid-top doublet for a middle spacepoint is
    // larger than 0, the entry is added to the doublet counter
    if (n_mid_bot > 0 && n_mid_top > 0) {
        auto pos = atomicAdd(&num_compat_spM_per_bin, 1);
        atomicAdd(&num_mid_bot_per_bin, n_mid_bot);
        atomicAdd(&num_mid_top_per_bin, n_mid_top);
        doublet_counter_per_bin[pos] = {spM_loc, n_mid_bot, n_mid_top};
    }
}

}  // namespace cuda
}  // namespace traccc
